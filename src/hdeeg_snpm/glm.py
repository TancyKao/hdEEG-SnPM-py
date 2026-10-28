"""General linear model tier: design presets, per-channel t and F, Freedman-Lane permutation.

Ports ``snpm_glm_design.m``, ``snpm_glm_stat.m``, ``snpm_glm_fl_context.m``, the index use of
``snpm_glm_permute.m`` and ``snpm_perm_correction.m``.

Method: per-channel t (one-row contrast) or F (several rows), family-wise correction from a
max-statistic permutation null over TFCE and over cluster mass (Maris & Oostenveld 2007; Nichols
& Holmes 2001), with nuisance columns handled by Freedman-Lane residual permutation inside
exchangeability blocks (Winkler et al. 2014).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from scipy import stats

from . import cluster_stat
from . import perm as _perm
from .clusters import ChannelGraph, find_clusters
from .legacy import ZERO_SS_RTOL, _at_least, _fwe_p, _sort_desc
from .linalg import aliased_columns, rank_tol
from .tfce import tfce

PRESETS = ("anova1", "ancova", "regression", "rmanova", "mixed2way")
_EPS = np.finfo(float).eps

# Reasons a loop does not test a channel. The last two are shared with MATLAB word for word.
MISSING = "missing data: the channel is not complete across every analysed row"
NO_VARIATION = "no variation left after the nuisance model"
PERFECT_FIT = "perfect fit, statistic undefined"
# Why a whole loop is refused (shared with MATLAB word for word).
NO_ERROR_DF = "no error degrees of freedom left after fitting the design"


class NoErrorDF(ValueError):
    """The design leaves no error degrees of freedom (rows - rank(X) < 1)."""


class GLMInputError(ValueError):
    """Unusable GLM input; ``code`` is the suffix of MATLAB's ``core_snpm_glm:`` identifier
    (``effectAliased``, ``ancovaNoCovariate``, ``predictorNotNumeric``)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class GLMNuisanceRankDeficient(UserWarning):
    """MATLAB ``snpm:glmNuisanceRankDeficient``: the nuisance columns are linearly dependent but
    the effect is estimable, so the analysis runs."""


ANCOVA_NO_COVARIATE = "ANCOVA needs at least one covariate; with none, use anova1."


def covariate_list(cols) -> list[str]:
    """``covariate_cols`` as a list of names, blank entries dropped (None, '', [''] -> []) and
    the rest trimmed (' age ' -> 'age'), as ``snpm_glm_covariate_list.m``."""
    if cols is None:
        return []
    return [str(c).strip() for c in np.atleast_1d(cols) if str(c).strip()]


def require_covariate(preset: str, covariate_cols=None, design: Design | None = None):
    """``ancovaNoCovariate`` (``snpm_glm_require_covariate.m``): an ANCOVA without a covariate
    is a one-way ANOVA labelled covariate-adjusted. Without ``design`` (before the data are
    read) it refuses when ``covariate_cols`` names no column; with it, when the design kept no
    covariate column, every listed one being constant or one-level on these rows. Any other
    preset passes."""
    if preset.lower() != "ancova":
        return
    if design is not None:
        none = not any(nm.startswith("covariate ") for nm in design.col_names)
    else:
        none = not covariate_list(covariate_cols)
    if none:
        raise GLMInputError("ancovaNoCovariate", ANCOVA_NO_COVARIATE)


def require_numeric_predictor(column, name):
    """``predictorNotNumeric``: a regression slope needs a numeric predictor. Numeric columns
    pass. A text column is refused even when its values look like numbers, because
    ``read_table`` already reads a numeric column (blanks included) as numbers. A true/false
    column is refused too: MATLAB's readtable reads it as text, and a two-level predictor is a
    group comparison. The value quoted is the first one not missing or blank, as text; a
    boolean column is quoted as MATLAB prints a logical ('true', 'false'), text as read."""
    column = pd.Series(column)
    is_bool = pd.api.types.is_bool_dtype(column)
    if pd.api.types.is_numeric_dtype(column) and not is_bool:
        return
    first = next((str(v).strip() for v in column
                  if not pd.isna(v) and str(v).strip()), "")
    if is_bool:
        first = first.lower()
    raise GLMInputError(
        "predictorNotNumeric",
        f"The predictor column '{name}' is not numeric (it holds text such as '{first}'); "
        "regression needs a numeric predictor. To compare groups, use anova1 or ancova.")


def _require_error_df(X: np.ndarray, name: str):
    """Refuse a saturated design before any channel rule runs: with rows - rank(X) < 1 every
    channel would be a perfect fit, which is a property of the design, not of the data."""
    n, r = X.shape[0], np.linalg.matrix_rank(X)
    if n - r < 1:
        raise NoErrorDF(f"{name}: {NO_ERROR_DF} ({n} rows, design rank {r}). Add subjects or "
                        "remove covariates.")


# --------------------------------------------------------------------------- design

@dataclass
class Design:
    X: np.ndarray                 # (n_obs, p)
    C: np.ndarray                 # (q, p); q == 1 gives t, q > 1 gives F
    nuisance: np.ndarray          # (p,) bool, columns held fixed under Freedman-Lane
    blocks: np.ndarray            # (n_obs,) exchangeability-block id
    perm_type: str                # 'free' or 'within'
    effect_label: str
    # Pairwise comparisons that follow an effect with three or more levels: (label, a, b) for
    # mean(a) - mean(b), over the levels of ``factor_col``. Each is run as its own two-level
    # test on the rows of a and b only (see ``plan_preset``).
    posthoc: list[tuple[str, object, object]] = field(default_factory=list)
    factor_col: str | None = None
    # One name per column of X, for messages (snpm_glm_design.m col_names): '(Intercept)';
    # 'predictor <col>'; 'covariate <col>' or 'covariate <col>_<level>'; '<factor>_<level>';
    # '<group>_<l>:<condition>_<l>'.
    col_names: list[str] = field(default_factory=list)

    @property
    def contrast_type(self) -> str:
        return "t" if self.C.shape[0] == 1 else "F"


def _levels(x, order=None) -> tuple[list, np.ndarray]:
    """Levels in order of first appearance (or in ``order``, keeping those present) and the
    level index of each row."""
    codes, levels = pd.factorize(np.asarray(x), sort=False)
    if (codes < 0).any():
        raise ValueError("a design column contains missing values")
    levels = list(levels)
    if order is not None:
        wanted = [v for v in order if v in levels] + [v for v in levels if v not in order]
        codes = np.array([wanted.index(levels[k]) for k in codes])
        levels = wanted
    return levels, codes


def _ref_dummy(x, order=None) -> tuple[np.ndarray, list]:
    """Reference-coded dummies, first level as reference: (n, k - 1)."""
    levels, codes = _levels(x, order)
    return (codes[:, None] == np.arange(1, len(levels))).astype(float), levels


def _no_missing(column, name):
    bad = np.flatnonzero(pd.isna(column).to_numpy())
    if bad.size:
        raise ValueError(f"Design column '{name}' has a missing value in row(s) "
                         f"{', '.join(str(i + 1) for i in bad[:10])}"
                         f"{' ...' if bad.size > 10 else ''}. Remove those rows or fill the value; "
                         "a row without a design value cannot enter the model.")


def _covariates(meta: pd.DataFrame, cols) -> tuple[np.ndarray, list[str]]:
    """Numeric covariates as they are; other columns dummy-coded with the alphabetically first
    level as reference. Every column is mean-centred. Also returns one name per column
    ('covariate <col>' or 'covariate <col>_<level>')."""
    if cols is None or len(cols) == 0:
        return np.empty((len(meta), 0)), []
    if isinstance(cols, str):
        cols = [cols]
    parts, names = [], []
    for c in cols:
        v = meta[c]
        _no_missing(v, c)
        if pd.api.types.is_numeric_dtype(v) or pd.api.types.is_bool_dtype(v):
            parts.append(v.to_numpy(dtype=float)[:, None])
            names.append(f"covariate {c}")
        else:
            cats = sorted(v.dropna().astype(str).unique())
            parts.append((v.astype(str).to_numpy()[:, None] == np.array(cats[1:])).astype(float))
            names += [f"covariate {c}_{lv}" for lv in cats[1:]]
    cov = np.concatenate(parts, axis=1)
    # A covariate that does not vary over these rows is the intercept again; it carries nothing.
    if cov.size:
        varies = cov.max(axis=0) > cov.min(axis=0)
        cov, names = cov[:, varies], [nm for nm, k in zip(names, varies) if k]
    return cov - cov.mean(axis=0), names


def _num2str(v) -> str:
    """A level as ``snpm_glm_design.m`` ``tostr`` writes it: a logical as ``false``/``true``
    (MATLAB ``string``), a number as ``num2str`` (whole numbers without a decimal point)."""
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    v = float(v)
    if v.is_integer():
        return str(int(v))
    digits = max(int(np.ceil(np.log10(abs(v)))), 1) + 4
    return f"{v:.{digits}g}"


def _level_names(col, levels) -> list[str]:
    """'<col>_<level>' for every non-reference level, as _ref_dummy codes them."""
    return [f"{col}_{_num2str(lv)}" for lv in levels[1:]]


def _select(p: int, cols) -> np.ndarray:
    C = np.zeros((len(cols), p))
    C[np.arange(len(cols)), cols] = 1
    return C


def _pairwise(levels) -> list[tuple[str, object, object]]:
    """Every pair of levels, in level order: (label, a, b) meaning mean(a) - mean(b)."""
    return [(f"{_label(a)} vs {_label(b)}", a, b)
            for i, a in enumerate(levels) for b in levels[i + 1:]]


def _label(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float, np.integer, np.floating)) else str(v)


def build_design(preset: str, meta: pd.DataFrame, *, group_col=None, condition_col=None,
                 subject_col=None, predictor_col=None, covariate_cols=None,
                 effect: str = "interaction", condition_levels=None) -> Design:
    """Turn a preset and a column mapping into the design, contrast and permutation scheme.

    ``mixed2way`` builds one within-subject effect at a time (``effect='interaction'`` or
    ``'condition'``); the between-group main effect is a separate test on subject means.

    The ``mixed2way`` condition effect is tested in the full model. Group enters the
    group-by-condition columns coded to sum to zero (the first group is -1 in every column), so
    the condition coefficients are the condition effect averaged over groups with equal weight,
    which is what a standard mixed ANOVA reports, and a real interaction sits in the nuisance
    model instead of in the residuals that are permuted.
    """
    preset = preset.lower()
    if preset == "regression" and predictor_col in meta:
        require_numeric_predictor(meta[predictor_col], predictor_col)
    for role in (group_col, condition_col, subject_col, predictor_col):
        if role is not None and role in meta:
            _no_missing(meta[role], role)
    n = len(meta)
    ones = np.ones((n, 1))
    cov, cov_names = _covariates(meta, covariate_cols)
    posthoc: list = []
    factor = None

    if preset in ("anova1", "ancova"):
        Dg, levels = _ref_dummy(meta[group_col])
        X = np.hstack([ones, Dg] + ([cov] if preset == "ancova" else []))
        names = (["(Intercept)"] + _level_names(group_col, levels)
                 + (cov_names if preset == "ancova" else []))
        interest = np.arange(1, 1 + Dg.shape[1])
        perm_type, blocks = "free", np.ones(n, dtype=int)
        label = f"group ({len(levels)} levels)" + (", covariate-adjusted" if preset == "ancova" else "")
        posthoc, factor = _pairwise(levels), group_col
    elif preset == "regression":
        pred = meta[predictor_col].to_numpy(dtype=float)
        X = np.hstack([ones, (pred - np.nanmean(pred))[:, None], cov])
        names = ["(Intercept)", f"predictor {predictor_col}"] + cov_names
        interest = np.array([1])
        perm_type, blocks = "free", np.ones(n, dtype=int)
        label = f"slope of {predictor_col}"
    elif preset in ("rmanova", "mixed2way"):
        Ds, slevels = _ref_dummy(meta[subject_col])
        # ``condition_levels`` fixes the level order from outside, so that the condition effect
        # inside one group lists and signs its pairs as the whole table does.
        Dc, clevels = _ref_dummy(meta[condition_col], condition_levels)
        names = (["(Intercept)"] + _level_names(subject_col, slevels)
                 + _level_names(condition_col, clevels))
        if preset == "mixed2way":
            glevels = _levels(meta[group_col])[0]
            names += [f"{g}:{c}" for g in _level_names(group_col, glevels)
                      for c in _level_names(condition_col, clevels)]   # group outer
        ns, nc = Ds.shape[1], Dc.shape[1]
        perm_type, blocks = "within", _levels(meta[subject_col])[1]
        if preset == "mixed2way" and effect.lower() == "interaction":
            Dg, _ = _ref_dummy(meta[group_col])
            Dx = np.hstack([Dg[:, [a]] * Dc for a in range(Dg.shape[1])])
            X = np.hstack([ones, Ds, Dc, Dx])
            interest = np.arange(X.shape[1] - Dx.shape[1], X.shape[1])
            label = "two-way mixed ANOVA (group x condition)"
        elif preset == "mixed2way" and effect.lower() != "condition":
            raise ValueError("mixed2way builds 'interaction' or 'condition' here; the group main "
                             "effect is a between-subject test on subject means (anova1).")
        else:
            X = np.hstack([ones, Ds, Dc])
            if preset == "mixed2way":
                Dg, _ = _ref_dummy(meta[group_col])
                Eg = Dg - (Dg.sum(axis=1, keepdims=True) == 0)      # sum-to-zero group coding
                X = np.hstack([X] + [Eg[:, [a]] * Dc for a in range(Eg.shape[1])])
            interest = np.arange(1 + ns, 1 + ns + nc)
            label = (f"condition ({len(clevels)} levels), within-subject" if preset == "rmanova"
                     else f"condition main effect ({len(clevels)} levels)")
            posthoc, factor = _pairwise(clevels), condition_col
    else:
        raise ValueError(f"Unknown preset: {preset}")

    if len(interest) < 2:
        posthoc = []          # two levels: the main effect already is the one pairwise contrast
    nuisance = np.ones(X.shape[1], dtype=bool)
    nuisance[interest] = False
    return Design(X=X, C=_select(X.shape[1], interest), nuisance=nuisance, blocks=blocks,
                  perm_type=perm_type, effect_label=label, posthoc=posthoc, factor_col=factor,
                  col_names=names)


@dataclass
class Estimability:
    estimable: bool
    nuisance_deficient: bool
    effect_items: list[str]
    nuisance_items: list[str]


def estimability(design: Design, enforce: bool = False) -> Estimability:
    """Is the effect estimable apart from its Freedman-Lane nuisance (``snpm_glm_estimability.m``)?

    With Z the nuisance columns and Xe the effect columns, and ``tol = max(X.shape) * eps *
    s_max(X)`` from the full X, the identity contrast on Xe is estimable exactly when
    ``rank([Z Xe], tol) == rank(Z, tol) + cols(Xe)``. When it is not, the pinv solution still
    gives a finite statistic and every p-value would be arbitrary: ``enforce`` refuses
    (``effectAliased``), even when only one effect column is aliased. When only Z is
    rank-deficient the contrast is estimable and the Freedman-Lane fit is the unique projection,
    so ``enforce`` warns (:class:`GLMNuisanceRankDeficient`) and the analysis runs.

    Aliased columns are named by a left-to-right walk over Z first, then Xe. X is fixed across
    Freedman-Lane permutations, so the check on the observed design is exact. Texts are
    MATLAB's word for word."""
    X = design.X
    nuis = np.asarray(design.nuisance, bool)
    names = list(design.col_names)
    if len(names) != X.shape[1]:
        names = [f"column {j + 1}" for j in range(X.shape[1])]
    info = Estimability(True, False, [], [])
    if X.size == 0:
        return info
    tol = rank_tol(X)
    order = list(np.flatnonzero(nuis)) + list(np.flatnonzero(~nuis))
    rZ = np.linalg.matrix_rank(X[:, nuis], tol) if nuis.any() else 0
    info.estimable = bool(np.linalg.matrix_rank(X[:, order], tol) == rZ + int((~nuis).sum()))
    info.nuisance_deficient = bool(rZ < nuis.sum())
    if not info.estimable or info.nuisance_deficient:
        _, items = aliased_columns(X, names, order, tol, "is zero on every row")
        for j, item in items:
            (info.nuisance_items if nuis[j] else info.effect_items).append(item)
    if not enforce:
        return info
    if not info.estimable:
        raise GLMInputError(
            "effectAliased",
            f"The effect tested ({design.effect_label}) cannot be separated from the nuisance "
            f"columns of the design: {'; '.join(info.effect_items)}. Its contrast is not "
            "estimable, so the statistic and every p-value would be arbitrary. Remove the "
            "covariate that repeats the effect, or test a different effect.")
    if info.nuisance_deficient:
        warnings.warn(
            "The nuisance columns of the design are linearly dependent: "
            f"{'; '.join(info.nuisance_items)}. The effect is still estimable and the analysis "
            "runs; the adjustment is for the space these columns span, which the redundant "
            "column does not enlarge. Check the covariate list.",
            GLMNuisanceRankDeficient, stacklevel=3)
    return info


# --------------------------------------------------------------------------- statistic

def _pinv(a: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(a, max(a.shape) * _EPS)


def _ols(Y, X, C):
    q = C.shape[0]
    xtx_inv = _pinv(X.T @ X)
    B = xtx_inv @ (X.T @ Y)
    rss = ((Y - X @ B) ** 2).sum(axis=0)
    df2 = Y.shape[0] - np.linalg.matrix_rank(X)
    sigma2 = rss / df2
    CB = C @ B
    M = C @ xtx_inv @ C.T
    with np.errstate(invalid="ignore", divide="ignore"):
        if q == 1:
            stat = (CB / np.sqrt(sigma2 * M)).ravel()
            p = 2 * stats.t.cdf(-np.abs(stat), df2)
        else:
            stat = ((_pinv(M) @ CB) * CB).sum(axis=0) / q / sigma2
            p = stats.f.sf(stat, q, df2)
    # A zero error term: the statistic is defined as 0 (p = 1), at every labelling alike.
    undefined = rss <= ZERO_SS_RTOL * (Y ** 2).sum(axis=0)
    return np.where(undefined, 0.0, stat), np.where(undefined, 1.0, p), df2, undefined


def glm_stat(Y: np.ndarray, X: np.ndarray, C: np.ndarray, return_undefined: bool = False):
    """Ordinary least squares for every channel at once; returns (stat, p, df1, df2).

    ``stat`` is a signed t for a one-row contrast and F otherwise. Channels with some missing
    rows are fitted on the rows they have (left NaN if under-determined); all-NaN channels stay
    NaN. Where the residual sum of squares is zero (``SS <= 1e-20 * sum(Y ** 2)``) the statistic
    is 0 and p is 1; ``return_undefined`` adds the mask of those channels as a fifth output.
    """
    Y = np.asarray(Y, dtype=float)
    C = np.atleast_2d(C)
    n_obs, n_ch = Y.shape
    stat = np.full(n_ch, np.nan)
    p = np.full(n_ch, np.nan)
    undefined = np.zeros(n_ch, dtype=bool)
    nan = np.isnan(Y)
    has_nan = nan.any(axis=0)
    full = np.flatnonzero(~has_nan)
    if full.size:
        stat[full], p[full], df2, undefined[full] = _ols(Y[:, full], X, C)
    else:
        df2 = n_obs - np.linalg.matrix_rank(X)
    for ch in np.flatnonzero(has_nan & ~nan.all(axis=0)):
        rows = ~nan[:, ch]
        Xc = X[rows]
        if np.linalg.matrix_rank(Xc) < Xc.shape[1] or rows.sum() <= Xc.shape[1]:
            continue
        s, pp, _, u = _ols(Y[rows][:, [ch]], Xc, C)
        stat[ch], p[ch], undefined[ch] = s[0], pp[0], u[0]
    if return_undefined:
        return stat, p, C.shape[0], df2, undefined
    return stat, p, C.shape[0], df2


# --------------------------------------------------------------------------- permutation

@dataclass
class FLContext:
    """Nuisance fit and residuals, computed once and reused by every permutation."""

    fit: np.ndarray        # (n_obs, n_ch) nuisance fitted values
    resid: np.ndarray      # (n_obs, n_ch) residuals
    blocks: np.ndarray
    perm_type: str
    evaluable: np.ndarray  # (n_ch,) channels finite in every permutation, with something to permute
    novar: np.ndarray | None = None   # (n_ch,) finite, but no variation left after the nuisance model


def fl_context(Y: np.ndarray, design: Design) -> FLContext:
    """Fit the nuisance columns once. A channel with any missing cell is not evaluable: its
    permuted values would be NaN in every permutation, so it is dropped from the observed map
    too and both are defined on one channel set.

    Nor is a channel whose nuisance residuals are zero (``SS <= 1e-20 * sum(Y ** 2)``): every
    permuted map would equal the observed one, so there is nothing to permute."""
    Y = np.asarray(Y, dtype=float)
    Z = design.X[:, design.nuisance]
    if Z.shape[1] == 0:
        Z = np.ones((Y.shape[0], 1))
    fit = np.full(Y.shape, np.nan)
    ok = np.isfinite(Y).all(axis=0)
    if ok.any():
        beta = np.linalg.lstsq(Z, Y[:, ok], rcond=None)[0]
        fit[:, ok] = Z @ beta
    resid = Y - fit
    finite = np.isfinite(fit).all(axis=0) & np.isfinite(resid).all(axis=0)
    with np.errstate(invalid="ignore"):
        novar = finite & ((resid ** 2).sum(axis=0) <= ZERO_SS_RTOL * (Y ** 2).sum(axis=0))
    return FLContext(fit, resid, design.blocks, design.perm_type, finite & ~novar, novar)


@dataclass
class MassCluster:
    channels: np.ndarray
    mass: float
    p: float
    threshold: float
    permutations: int


@dataclass
class CorrectionResult:
    real_T: np.ndarray
    real_TFCE: np.ndarray
    tMaxTFCE: np.ndarray          # sorted descending
    maxMass: np.ndarray           # sorted descending
    critical_T_TFCE: float
    p_real: np.ndarray
    p_correctedTFCE: np.ndarray
    clusters: list[MassCluster]
    excluded_channels: np.ndarray
    n_perm: int
    perms: np.ndarray | None = field(default=None, repr=False)
    excluded_reasons: list[str] = field(default_factory=list)   # one per excluded channel
    cluster_statistic: str = cluster_stat.DEFAULT                  # 'sum' or 'mean'

    @property
    def excluded_mask(self) -> np.ndarray:
        m = np.zeros(self.real_T.size, dtype=bool)
        m[self.excluded_channels] = True
        return m


def _p_clusters(p_model, wald, alpha, graph, rule=cluster_stat.DEFAULT):
    """Connected channels with model p < alpha, each scored by ``cluster_stat.score``: the sum
    of the Wald statistic over the cluster, divided by the number of channels under 'mean'."""
    with np.errstate(invalid="ignore"):
        mask = (p_model < alpha).astype(float)
    return [(c, cluster_stat.score(wald, c, rule)) for c in find_clusters(mask, 0.5, graph)]


def perm_correction(real_stat, real_p, perm_stat_fn, graph: ChannelGraph, n_perm: int,
                    contrast_type: str, evaluable=None, alpha: float = 0.05, E: float = 0.5,
                    H: float = 2.0, dh: float = 0.1, undefined=None,
                    reasons=None, cluster_statistic: str = cluster_stat.DEFAULT
                    ) -> CorrectionResult:
    """TFCE max-statistic null and cluster-mass null from one permutation loop.

    ``cluster_statistic`` ('sum' or 'mean') is how a cluster is scored; TFCE does not use it.

    ``perm_stat_fn(i)`` returns ``(stat, p)`` for permutation ``i``. Channels outside
    ``evaluable`` are NaN in the observed map and in every permuted map. Channels in
    ``undefined`` (observed statistic undefined) are NaN in the observed map only and stay in
    every permuted map, which keeps the test exact. ``reasons`` gives, per channel, why a
    channel outside ``evaluable`` was left out (default: missing data).
    """
    is_f = contrast_type.upper() == "F"
    rule = cluster_stat.resolve(cluster_statistic)
    real_stat = np.array(real_stat, dtype=float)
    real_p = np.array(real_p, dtype=float)
    evaluable = (np.ones(graph.n, bool) if evaluable is None else np.asarray(evaluable, bool))
    if evaluable.size != graph.n or real_stat.size != graph.n:
        raise ValueError("statistic map, evaluable mask and channel graph must have the same size")
    undefined = (np.zeros(graph.n, bool) if undefined is None
                 else evaluable & np.asarray(undefined, bool))
    real_stat[~evaluable | undefined] = np.nan
    real_p[~evaluable | undefined] = np.nan

    def wald(s):
        return s if is_f else s ** 2

    real_tfce = tfce(real_stat, graph, E, H, dh)
    real_clusters = _p_clusters(real_p, wald(real_stat), alpha, graph, rule)
    max_tfce = np.zeros(n_perm)
    max_mass = np.zeros(n_perm)
    for i in range(n_perm):
        s, pp = perm_stat_fn(i)
        s = np.where(evaluable, s, np.nan)
        pp = np.where(evaluable, pp, np.nan)
        enhanced = tfce(s, graph, E, H, dh)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_tfce[i] = np.nanmax(enhanced if is_f else np.abs(enhanced))
        found = _p_clusters(pp, wald(s), alpha, graph, rule)
        if found:
            max_mass[i] = max(m for _, m in found)
    max_tfce = _sort_desc(max_tfce)
    max_mass = _sort_desc(max_mass)
    p_tfce = _fwe_p(real_tfce, max_tfce, two_sided=not is_f)
    p_tfce[~evaluable | undefined] = np.nan
    if real_clusters:
        clusters = [MassCluster(c, m, (_at_least(max_mass, m).sum() + 1) / (n_perm + 1), alpha, n_perm)
                    for c, m in real_clusters]
    else:
        clusters = [MassCluster(np.empty(0, dtype=int), 0.0, 1.0, alpha, n_perm)]
    return CorrectionResult(
        real_T=real_stat, real_TFCE=real_tfce, tMaxTFCE=max_tfce, maxMass=max_mass,
        critical_T_TFCE=max_tfce[int(np.floor(alpha * n_perm))], p_real=real_p,
        p_correctedTFCE=p_tfce, clusters=clusters,
        excluded_channels=np.flatnonzero(~evaluable | undefined), n_perm=n_perm,
        cluster_statistic=rule,
        excluded_reasons=[PERFECT_FIT if undefined[i] else
                          (MISSING if reasons is None else str(reasons[i]))
                          for i in np.flatnonzero(~evaluable | undefined)])


def run_effect(Y, design: Design, graph: ChannelGraph, n_perm: int = 5000, C=None,
                rng: np.random.Generator | None = None, perms=None, alpha: float = 0.05,
                E: float = 0.5, H: float = 2.0, ctx: FLContext | None = None,
                cluster_statistic: str = cluster_stat.DEFAULT) -> CorrectionResult:
    """Observed map plus Freedman-Lane permutation correction for one contrast.

    ``C`` defaults to the design's own contrast. ``perms`` is an (n_perm, n_obs) row-index matrix used in place of a draw.
    """
    Y = np.asarray(Y, dtype=float)
    C = design.C if C is None else np.atleast_2d(C)
    ctx = fl_context(Y, design) if ctx is None else ctx
    if perms is None:
        rng = rng if rng is not None else np.random.default_rng()
        perms = _perm.freedman_lane_indices(Y.shape[0], n_perm, design.perm_type,
                                            design.blocks, rng)
    perms = np.asarray(perms, dtype=int)
    stat, p, _, _, undefined = glm_stat(Y, design.X, C, return_undefined=True)

    def perm_stat(i):
        s, pp, _, _ = glm_stat(ctx.fit + ctx.resid[perms[i]], design.X, C)
        return s, pp

    novar = ctx.novar if ctx.novar is not None else np.zeros(Y.shape[1], bool)
    out = perm_correction(stat, p, perm_stat, graph, perms.shape[0],
                          "t" if C.shape[0] == 1 else "F", ctx.evaluable, alpha, E, H,
                          undefined=undefined,
                          reasons=np.where(novar, NO_VARIATION, MISSING).astype(object),
                          cluster_statistic=cluster_statistic)
    out.perms = perms
    return out


# --------------------------------------------------------------------------- presets

@dataclass
class Loop:
    """One permutation loop of a preset: the rows it uses, its design and its contrast."""

    role: str                 # 'main', 'group', 'condition', 'interaction', 'condition@<group>',
                              # with '/posthoc:<label>' appended for a post-hoc contrast
    Y: np.ndarray
    rows: np.ndarray          # rows of the input table behind each row of Y
    design: Design
    C: np.ndarray | None
    result: CorrectionResult | None = None
    mask: np.ndarray | None = None   # channels this loop may test, fixed before any permutation
    units: str = ""                  # for a pair: what it was run on, in words
    dropped: list[str] = field(default_factory=list)       # subjects left out of this pair
    skipped: list[tuple[str, str]] = field(default_factory=list)   # on an effect: (pair, reason)

    @property
    def contrast_type(self) -> str:
        return "t" if self.C.shape[0] == 1 else "F"

    @property
    def global_mask(self) -> np.ndarray:
        """The channels the whole-head average is taken over: the finiteness mask only, so the
        mixed2way whole-head tests share one channel set whatever each loop excludes."""
        finite = np.isfinite(self.Y).all(axis=0)
        return finite if self.mask is None else finite & self.mask

    def global_test(self):
        """Parametric test of this contrast on the channel average: (statistic, p, df1, df2,
        reason). With a zero error term the true statistic is infinite, so statistic and p are
        NaN and ``reason`` says why; the permutation rule of 0 at a zero error term does not
        apply to a parametric p-value. ``reason`` is '' otherwise."""
        keep = self.global_mask
        if not keep.any():
            return np.nan, np.nan, np.nan, np.nan, ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            yg = np.nanmean(self.Y[:, keep], axis=1)[:, None]
        stat, p, df1, df2, undefined = glm_stat(yg, self.design.X, self.C, return_undefined=True)
        if undefined[0]:
            return np.nan, np.nan, df1, df2, PERFECT_FIT
        return float(stat[0]), float(p[0]), df1, df2, ""


def _pair_loop(parent: Loop, preset: str, meta: pd.DataFrame, label, a, b, cols) -> Loop | str:
    """The comparison of levels a and b as its own two-level test on their rows only.

    Under the hypothesis that a and b do not differ, only the a and b rows can be exchanged, so
    the other levels take no part: they are not fitted and not permuted. When the pair cannot
    be tested the reason is returned instead, and is reported with the effect.
    """
    design = parent.design
    factor = meta[design.factor_col].to_numpy()
    keep = (factor == a) | (factor == b)
    name = f"{parent.role}/posthoc:{label}"
    dropped: list[str] = []
    if design.perm_type == "within":
        ids = meta[cols["subject_col"]].to_numpy()
        present = pd.unique(ids[keep])
        both = [s for s in present if (factor[keep & (ids == s)] == a).any()
                and (factor[keep & (ids == s)] == b).any()]
        dropped = [str(s) for s in present if s not in set(both)]
        if dropped:
            warnings.warn(f"{name}: subject(s) {', '.join(dropped)} have only one of the two "
                          "levels and are left out of this comparison.", stacklevel=3)
        keep &= np.isin(ids, both)
        if len(both) < 2:
            return "fewer than 2 subjects have both levels"
        units = f"{len(both)} of {len(pd.unique(ids))} subjects"
        if len(both) < 6:
            units += f" (smallest possible two-sided p-value {2 / 2 ** len(both):.3g})"
            warnings.warn(f"{name}: only {len(both)} subjects have both levels, so the smallest "
                          f"possible two-sided p-value is {2 / 2 ** len(both):.3g}.", stacklevel=3)
    else:
        na, nb = int((factor[keep] == a).sum()), int((factor[keep] == b).sum())
        units = f"{na} and {nb} subjects"
        if min(na, nb) < 4:
            warnings.warn(f"{name}: only {min(na, nb)} in the smaller group.", stacklevel=3)
    sel = np.flatnonzero(keep)
    sub = meta.iloc[sel].reset_index(drop=True)
    kw = dict(cols)
    if preset == "mixed2way":
        kw["effect"] = "condition"
    d = build_design(preset, sub, **kw)
    # Estimability of the pair's effect apart from its nuisance, the rule the main design is
    # refused by: a pair whose only dependency is a redundant covariate runs, as the main design
    # does.
    if d.C.shape[0] != 1 or not estimability(d).estimable:
        return ("the design for these two levels cannot be estimated (a covariate may coincide "
                "with the two groups)")
    if d.X.shape[0] - np.linalg.matrix_rank(d.X, rank_tol(d.X)) < 1:
        return "too few rows for these two levels to leave any error degrees of freedom"
    # The design's own contrast is (its second level) - (its reference level); report a - b.
    # The reference is the first level in the order the builder used: a fixed condition order
    # when one was given, otherwise the first to appear in these rows.
    order = kw.get("condition_levels") if design.factor_col == cols.get("condition_col") else None
    first = _levels(sub[design.factor_col], order)[0][0]
    return Loop(name, parent.Y[sel], parent.rows[sel], d, d.C * (-1.0 if first == a else 1.0),
                mask=parent.mask, units=units, dropped=dropped)


def _with_posthoc(role, preset, Y, rows, meta, mask=None, **cols) -> list[Loop]:
    """An effect, then one two-level test per pair of levels when it has three or more."""
    meta = meta.reset_index(drop=True)
    kw = dict(cols)
    if preset == "mixed2way":
        kw["effect"] = role
    Y = np.asarray(Y, dtype=float)
    mask = np.isfinite(Y).all(axis=0) if mask is None else mask
    parent = Loop(role, Y, np.asarray(rows), build_design(preset, meta, **kw), None, mask=mask)
    # every listed covariate constant on these rows -> dropped -> no covariate left
    require_covariate(preset, cols.get("covariate_cols"), parent.design)
    # An effect aliased with the nuisance is refused; a nuisance-only dependency warns and runs.
    estimability(parent.design, enforce=True)
    _require_error_df(parent.design.X, role)
    parent.C = parent.design.C
    pairs = [] if role == "interaction" else parent.design.posthoc
    out = [parent]
    for label, a, b in pairs:
        got = _pair_loop(parent, preset, meta, label, a, b, cols)
        if isinstance(got, str):
            warnings.warn(f"{role}/posthoc:{label} skipped: {got}.", stacklevel=3)
            parent.skipped.append((label, got))
        else:
            out.append(got)
    return out


SIMPLE_EFFECT = "simple effect: condition within "


def _require_balance(meta, g, c, s):
    """Every subject needs exactly one row per condition and one group. A subject short of a
    condition would have its mean taken over fewer conditions, which biases the group effect
    whenever conditions differ."""
    conditions = pd.unique(meta[c])
    counts = meta.groupby([s, c], sort=False).size().unstack(fill_value=0).reindex(
        columns=conditions, fill_value=0)
    bad = counts.index[(counts != 1).any(axis=1)]
    if len(bad):
        detail = "; ".join(
            f"{sid} (" + ", ".join(f"{_label(k)}: {int(n)} rows" for k, n in counts.loc[sid].items()
                                   if n != 1) + ")" for sid in bad[:10])
        raise ValueError(
            f"mixed2way needs every subject to have exactly one row per condition. "
            f"{len(bad)} subject(s) do not: {detail}{' ...' if len(bad) > 10 else ''}. Remove "
            "them or complete the data; unbalanced data need a mixed model.")
    if (meta.groupby(s, sort=False)[g].nunique() != 1).any():
        raise ValueError("mixed2way: a subject appears in more than one group")


def plan_preset(preset: str, Y, meta: pd.DataFrame, **cols) -> list[Loop]:
    """Every permutation loop a preset runs, in run order.

    Single-effect presets give the main effect and, with three or more levels, one two-level
    test per pair of levels. ``mixed2way`` gives the group effect on subject means, the
    condition main effect, the interaction, then the condition effect inside each group. All
    ``mixed2way`` effects are tested on one channel set: the channels present in every row.

    A design with no error degrees of freedom is refused (:class:`NoErrorDF`). The ``mixed2way``
    effects have (conditions - 1) * (subjects - groups) of them (the group effect subjects -
    groups), so they are refused together, when every group has one subject. A simple effect
    in a group of n >= 2 subjects has (conditions - 1) * (n - 1) >= 1, so the one-subject skip
    below is the only case where a simple effect has none.
    """
    Y = np.asarray(Y, dtype=float)
    meta = meta.reset_index(drop=True)
    every = np.arange(len(meta))
    preset = preset.lower()
    if preset != "mixed2way":
        return _with_posthoc("main", preset, Y, every, meta, **cols)

    g, c, s = cols["group_col"], cols["condition_col"], cols["subject_col"]
    for col in (g, c, s):
        _no_missing(meta[col], col)
    _require_balance(meta, g, c, s)
    mask = np.isfinite(Y).all(axis=0)
    conditions = _levels(meta[c])[0]      # one condition order for every part of the report
    subjects, code = _levels(meta[s])
    first = np.array([np.flatnonzero(code == k)[0] for k in range(len(subjects))])
    means = np.vstack([Y[code == k].mean(axis=0) for k in range(len(subjects))])
    loops = _with_posthoc("group", "anova1", means, first, meta.iloc[first], mask, group_col=g)
    for effect in ("condition", "interaction"):
        loops += _with_posthoc(effect, "mixed2way", Y, every, meta, mask, **cols)
    interaction = next(lp for lp in loops if lp.role == "interaction")
    for level in _levels(meta[g])[0]:
        rows = np.flatnonzero((meta[g] == level).to_numpy())
        n_subj = len(pd.unique(meta[s].to_numpy()[rows]))
        if n_subj < 2:
            # Count subjects, not rows: one subject has several rows but no error term.
            why = f"only {n_subj} subject in this group; a within-group test needs at least 2"
            warnings.warn(f"Simple effect of condition within {_label(level)} skipped: {why}.",
                          stacklevel=2)
            interaction.skipped.append((f"{SIMPLE_EFFECT}{_label(level)}", why))
            continue
        loops += _with_posthoc(f"condition@{_label(level)}", "rmanova", Y[rows], rows,
                               meta.iloc[rows], mask, condition_col=c, subject_col=s,
                               condition_levels=conditions)
    return loops


def run_preset(preset: str, Y, meta: pd.DataFrame, graph: ChannelGraph, n_perm: int = 5000,
               rng: np.random.Generator | None = None, perms=None, alpha: float = 0.05,
               E: float = 0.5, H: float = 2.0, cluster_statistic: str = cluster_stat.DEFAULT,
               **cols) -> list[Loop]:
    """Run every loop of a preset (main effect, post-hoc pairs, mixed2way effects and simple
    effects), each scoring clusters by ``cluster_statistic``. ``perms`` is one row-index matrix
    per loop, in plan order; without it each loop draws fresh indices from ``rng``."""
    cluster_statistic = cluster_stat.resolve(cluster_statistic)
    loops = plan_preset(preset, Y, meta, **cols)
    if perms is not None and len(perms) != len(loops):
        raise ValueError(f"{len(loops)} permutation loops but {len(perms)} index matrices")
    rng = rng if rng is not None else np.random.default_rng()
    for k, loop in enumerate(loops):
        ctx = fl_context(loop.Y, loop.design)
        if loop.mask is not None:
            ctx = replace(ctx, evaluable=ctx.evaluable & loop.mask)
        loop.result = run_effect(loop.Y, loop.design, graph, n_perm, loop.C, rng,
                                 None if perms is None else perms[k], alpha, E, H, ctx,
                                 cluster_statistic)
    return loops
