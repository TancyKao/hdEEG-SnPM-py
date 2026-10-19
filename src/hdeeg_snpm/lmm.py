"""Mixed linear model tier: a per-channel linear mixed model with a random intercept per subject.

Ports ``snpm_lmm_fit.m``, ``snpm_lmm_permute_meta.m``, ``snpm_lmm_TFCE.m`` and
``snpm_lmm_cluster.m`` (Stephan et al. 2021).

The fit reproduces MATLAB ``fitlme`` with its defaults, as used by ``snpm_lmm_fit.m``: maximum
likelihood, reference (treatment) coding with the first sorted level as reference, residual
degrees of freedom ``N - rank(X)``, and the coefficient covariance ``sigma^2 inv(X' V^-1 X)``
taken at the fitted variance parameters (not the inverse of the full Hessian, which is what
statsmodels reports). A continuous effect is mapped as the t of its coefficient; a factor effect
as the marginal Wald F of its coefficients divided by their number, which is what
``anova(lme)`` reports.

Only one random structure is supported: a random intercept for one grouping variable,
``(1|Subject)``. Every other random part is refused (:class:`UnsupportedRandomEffects`). For that
model the likelihood with the fixed effects and the residual variance profiled out depends on one
number, ``theta`` = (subject SD) / (residual SD), and every channel's fit is a one-dimensional
minimisation done for all channels at once: a log grid that includes ``theta = 0`` brackets the
minimum, golden-section search narrows it, and Newton steps on the derivative finish it. There is
no iterative optimiser that can fail to converge, and a subject variance of zero (the boundary) is
found exactly. MATLAB reaches the same optimum to the tolerance of its own optimiser (gradient
1e-6), so agreement with MATLAB is about 1e-6 relative, not to rounding.

Failure handling follows ``snpm_lmm_fit.m``: a channel whose fit cannot be made (fewer than five
rows, a rank-deficient design) or whose statistic is not finite has statistic NaN, Wald 0 and
model p NaN. A permutation whose map has no finite statistic on any evaluable channel enters the
TFCE null as +Inf. TFCE and cluster correction draw their own permutations, as in MATLAB.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from . import perm as _perm
from .clusters import ChannelGraph, find_clusters
from .legacy import ZERO_SS_RTOL, _at_least, _fwe_p, _sort_desc
from .tfce import tfce

EFFECT_TYPES = ("continuous", "factor")
SCHEMES = ("within_subject", "group_label")
MIN_TRIALS = 5   # snpm_lmm_fit.m: no fit is attempted on fewer rows
# Word for word the reason core_snpm_lmm.m gives for an excluded channel.
MISSING = ("missing data: channel not complete across all analysed rows (trials), so the fitted "
           "row set would differ from the permuted one")

_GOLD = (np.sqrt(5.0) - 1.0) / 2.0
_GRID = np.concatenate([[0.0], np.logspace(-4, 4, 65)])
THETA_MAX = 1e3   # method-of-moments subject SD / residual SD at which a fit is degenerate
WITHIN_MIN = 1e-6  # within-subject share of variation at or below which a column is "constant"
# Word for word the reasons core_snpm_lmm.m gives (compared with MATLAB; do not reword).
CONST_RESPONSE = ("channel value constant within each subject: no within-subject variation for "
                  "the mixed model to estimate; analyse subject means with anova1/unpairedT")
CONST_PREDICTOR = ("predictor constant within each subject: its effect is between-subject only, "
                   "which the within-subject shuffle cannot test (use group_label or regression on "
                   "subject means)")
PERFECT_FIT = "perfect fit, statistic undefined"


class UnsupportedRandomEffects(ValueError):
    """The random part is anything other than one random intercept, ``(1|G)``."""


# --------------------------------------------------------------------------- specification

@dataclass
class LMMSpec:
    """The model, as ``spec`` in ``snpm_lmm_fit.m``.

    ``dv`` is a column of the metadata, or ``'POWER'`` when the channel value is the response.
    ``fixed`` references ``POWER`` and metadata columns in Wilkinson notation (``+``, ``*``,
    ``:``). ``random`` must be ``(1|G)``. ``perm`` defaults to ``group_label`` for a factor
    effect and ``within_subject`` for a continuous one.
    """

    dv: str
    fixed: str
    effect: str
    effect_type: str
    random: str = "(1|Subject)"
    subject: str = "Subject"
    group: str = "group"
    perm: str | None = None
    categorical: tuple[str, ...] | None = None

    def __post_init__(self):
        self.effect_type = self.effect_type.lower()
        if self.effect_type not in EFFECT_TYPES:
            raise ValueError("effect_type must be 'continuous' or 'factor'")
        if self.perm is None:
            self.perm = "group_label" if self.effect_type == "factor" else "within_subject"
        if self.perm not in SCHEMES:
            raise ValueError("perm must be 'within_subject' or 'group_label'")
        parse_random(self.random)

    @property
    def grouping(self) -> str:
        return parse_random(self.random)

    @property
    def formula(self) -> str:
        return f"{self.dv} ~ {self.fixed} + {self.random}"


def parse_random(random: str) -> str:
    """The grouping variable of ``(1|G)``; anything else raises."""
    m = re.fullmatch(r"\s*\(\s*1\s*\|\s*([A-Za-z]\w*)\s*\)\s*", random or "")
    if m is None:
        raise UnsupportedRandomEffects(
            f"Random part '{random}' is not supported. Only a random intercept for one grouping "
            "variable, such as '(1|Subject)', can be fitted; random slopes, several grouping "
            "variables and crossed random effects are not available in this version.")
    return m.group(1)


def _terms(fixed: str, columns=()) -> list[tuple[str, ...]]:
    """Fixed-effect terms (tuples of variable names) in fitlme's order.

    fitlme sees the metadata table with the channel appended as its last variable, ``POWER``, and
    orders terms by degree, then by the table position of their variables; the variables inside an
    interaction are in table order too. So ``POWER + group`` gives ``(Intercept), group_G2,
    group_G3, POWER``. Variables not in ``columns`` sort after ``POWER``, as written."""
    out: list[tuple[str, ...]] = []
    for piece in fixed.split("+"):
        piece = piece.strip()
        if piece in ("", "1"):
            continue
        if piece.startswith("-") or piece == "0":
            raise ValueError(f"Fixed part '{fixed}': removing the intercept is not supported")
        if "*" in piece:
            names = [v.strip() for v in piece.split("*")]
            new = [c for k in range(1, len(names) + 1) for c in combinations(names, k)]
        else:
            new = [tuple(v.strip() for v in piece.split(":"))]
        for t in new:
            if not all(re.fullmatch(r"[A-Za-z]\w*", v) for v in t):
                raise ValueError(f"Fixed part '{fixed}': cannot read term '{':'.join(t)}'")
            if not any(set(t) == set(u) for u in out):
                out.append(t)
    pos = {c: i for i, c in enumerate(columns)}
    pos.setdefault("POWER", len(pos))
    rank = lambda v: pos.get(v, len(pos) + 1)
    out = [tuple(sorted(t, key=rank)) for t in out]
    return sorted(out, key=lambda t: (len(t), [rank(v) for v in t]))


def _sorted_levels(x: pd.Series) -> list:
    """Levels as MATLAB ``categorical`` orders them: sorted values."""
    vals = pd.unique(x)
    try:
        return sorted(vals)
    except TypeError:
        return sorted(vals, key=str)


# --------------------------------------------------------------------------- missing metadata

class LMMInputError(ValueError):
    """Unusable mixed-model input; ``code`` is the MATLAB identifier's suffix."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def check_design(spec: LMMSpec, tail: str = "both", meta_columns=None):
    """Refuse, before any fit, the settings under which the permutation test cannot work.

    ``oneSidedTail``: the mixed model is two-sided only (a left tail would compare signed
    maxima, F has no side, and clusters always use the two-sided model p). ``effectNotPermuted``:
    under group_label the effect term must contain the group column, or relabelling subjects
    leaves every permuted map equal to the observed one. ``dvNotMetadata``: under within_subject
    the response must be a metadata column, since that is what is shuffled.
    The texts are MATLAB's (``core_snpm_lmm.m``), word for word; the tail check is case-sensitive.
    """
    if tail != "both":
        raise LMMInputError(
            "oneSidedTail",
            f"mixedmodel requires tail = 'both' (got '{tail}'). The effect map is a two-sided t or "
            "an F, and clusters are formed on the two-sided model p; a one-sided TFCE null of "
            "signed maxima would not match it.")
    if spec.perm == "group_label":
        # a token names the group column itself, or one of its coefficients (group_B)
        effect_vars = [v.strip() for v in re.split(r"[:*]", spec.effect)]
        if not any(v == spec.group or v.startswith(spec.group + "_") for v in effect_vars):
            raise LMMInputError(
                "effectNotPermuted",
                f"The group_label scheme relabels subjects between levels of '{spec.group}', but "
                f"the effect '{spec.effect}' does not contain '{spec.group}', so every permuted map "
                "would equal the observed one. Use within_subject for an effect that varies "
                "within subjects, or test an effect of the group column.")
    elif spec.dv == "POWER" or (meta_columns is not None and spec.dv not in meta_columns):
        raise LMMInputError(
            "dvNotMetadata",
            f"The within_subject scheme shuffles the response within each subject, so the "
            f"response must be a metadata column; '{spec.dv}' is not "
            + ("(it is the channel value)." if spec.dv == "POWER" else "among lmm_meta_cols.")
            + " Use group_label for a between-group effect on the channel value.")


def _is_missing(col: pd.Series) -> np.ndarray:
    """NaN, None, a missing category, or an empty / blank string."""
    miss = np.array(pd.isna(col), dtype=bool)   # a writable copy (pandas 3: copy-on-write)
    if not pd.api.types.is_numeric_dtype(col):
        text = col.astype("string").str.strip()
        miss = miss | np.array((text == "").fillna(True), dtype=bool)
    return miss


class LMMMetaNaNRows(UserWarning):
    """MATLAB ``snpm:lmmMetaNaNRows``: rows dropped for a missing model variable."""


class LMMSingleTrialSubject(UserWarning):
    """MATLAB ``snpm:lmmSingleTrialSubject``: a subject left with one trial (within_subject)."""


def _ids(x) -> list[str]:
    """An ID column as text, as MATLAB ``cellstr(string(x))``: whole numbers without '.0'."""
    out = []
    for v in pd.Series(x).tolist():
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        out.append(str(v))
    return out


def model_variables(spec: LMMSpec, columns) -> list[str]:
    """The variable set M of ``snpm_lmm_drop_missing_meta.m``: the response, every identifier in
    the fixed and random parts, the subject column, and the group column under group_label;
    POWER removed; kept only if it is a metadata column, in table order."""
    toks = re.findall(r"[A-Za-z]\w*", f"{spec.fixed} {spec.random}")
    m = {spec.dv, *toks, spec.subject} | ({spec.group} if spec.perm == "group_label" else set())
    m.discard("POWER")
    return [c for c in columns if c in m]


@dataclass
class MetaDrop:
    """``results_struct.lmm.dropped_rows``: rows dropped for a missing model variable."""

    n: int                                # rows dropped
    n_input: int                          # rows before the drop
    columns: list[str]                    # the variable set M, table order
    per_column: list[int]                 # missing cells per column of M
    rows: np.ndarray                      # 0-based indices into the table as read
    subjects_lost: list[str]              # subjects that lost every row
    n_subjects: int = 0                   # after the drop
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"n": self.n, "n_input": self.n_input, "columns": list(self.columns),
                "per_column": [int(v) for v in self.per_column],
                "rows": [int(i) for i in self.rows], "subjects_lost": list(self.subjects_lost),
                "n_subjects": self.n_subjects}


def dropped_text(d: MetaDrop) -> str:
    """One-line summary, ``snpm_lmm_dropped_text.m``."""
    if d is None or d.n == 0:
        return "Rows dropped for a missing model variable: none"
    parts = [f"{c} {k}" for c, k in zip(d.columns, d.per_column) if k > 0]
    txt = f"Rows dropped for a missing model variable: {d.n} of {d.n_input} ({', '.join(parts)})"
    if d.subjects_lost:
        txt += f"; subjects lost entirely: {', '.join(d.subjects_lost)}"
    return txt


def drop_missing_meta(meta: pd.DataFrame, spec: LMMSpec, power=None):
    """Drop every row with a missing model variable from the metadata and the channel matrix
    together, then check the design against the permutation scheme
    (``snpm_lmm_drop_missing_meta.m``). Returns ``(meta, power, MetaDrop)``.

    Missing = NaN, None, a missing category, or empty text. The checks run whether or not
    anything was dropped: fewer than 3 subjects (``tooFewSubjects``); under group_label a group
    varying within a subject (``groupVariesWithinSubject``) or fewer than 2 groups or a group with
    fewer than 2 subjects (``groupLevelTooFewSubjects``); under within_subject a subject with a
    single trial is a warning (``LMMSingleTrialSubject``).
    """
    meta = meta.reset_index(drop=True)
    cols = [str(c) for c in meta.columns]
    if spec.subject not in cols:
        raise LMMInputError("noSubjectColumn", f"The subject column '{spec.subject}' is not among "
                            "the metadata columns (lmm_meta_cols).")
    grp_perm = spec.perm == "group_label"
    if grp_perm and spec.group not in cols:
        raise LMMInputError("noGroupColumn", f"group_label permutation needs the group column "
                            f"'{spec.group}' among the metadata columns.")
    M = model_variables(spec, cols)
    miss = np.column_stack([_is_missing(meta[c]) for c in M])
    drop = miss.any(axis=1)
    n_in = len(meta)
    subj_col = meta[spec.subject]
    before = list(dict.fromkeys(_ids(subj_col[~_is_missing(subj_col)])))
    out = meta.loc[~drop].reset_index(drop=True)
    pw = None if power is None else np.asarray(power)[~drop]
    sid = _ids(out[spec.subject])
    after = list(dict.fromkeys(sid))
    info = MetaDrop(int(drop.sum()), n_in, M, [int(k) for k in miss.sum(axis=0)],
                    np.flatnonzero(drop), [s for s in before if s not in set(after)])
    if info.n > 0:
        parts = ", ".join(f"{c} {k}" for c, k in zip(M, info.per_column) if k > 0)
        lost = ", ".join(info.subjects_lost) or "none"
        msg = (f"{info.n} of {n_in} rows (trials) dropped for a missing model variable "
               f"({parts}); subjects that lost every trial: {lost}. The dropped rows are left out "
               "of the observed fit and every permutation alike; this assumes the missingness is "
               "unrelated to the effect tested.")
        info.notes.append(msg)
        warnings.warn(msg, LMMMetaNaNRows, stacklevel=2)
    nsub = len(after)
    info.n_subjects = nsub
    if nsub < 3:
        raise LMMInputError("tooFewSubjects", f"Only {nsub} subject(s) left after dropping rows "
                            "with missing model variables; at least 3 are needed.")
    if grp_perm:
        g = _ids(out[spec.group])
        per_subj: dict[str, set] = {}
        for si, gi in zip(sid, g):
            per_subj.setdefault(si, set()).add(gi)
        bad = [si for si in after if len(per_subj[si]) > 1]
        if bad:
            raise LMMInputError(
                "groupVariesWithinSubject",
                "group_label permutation relabels whole subjects, so the group must be constant "
                f"within each subject; it varies within {len(bad)} subject(s): "
                f"{', '.join(bad[:10])}.")
        levels = sorted(set(g))
        first = {}
        for si, gi in zip(sid, g):
            first.setdefault(si, gi)
        counts = [sum(1 for v in first.values() if v == lv) for lv in levels]
        if len(levels) < 2 or min(counts) < 2:
            raise LMMInputError(
                "groupLevelTooFewSubjects",
                "group_label permutation needs at least 2 groups with at least 2 subjects each; "
                "subjects per group after the drop: "
                f"{', '.join(f'{lv} {n}' for lv, n in zip(levels, counts))}.")
    elif spec.perm == "within_subject":
        n_tr = pd.Series(sid).value_counts(sort=False)
        single = [si for si in after if n_tr[si] == 1]
        if single:
            msg = (f"{len(single)} subject(s) have a single trial ({', '.join(single)}): the "
                   "within-subject shuffle cannot move their DV, so they add no permutation "
                   "variability.")
            info.notes.append(msg)
            warnings.warn(msg, LMMSingleTrialSubject, stacklevel=2)
    return out, pw, info


# --------------------------------------------------------------------------- model

class LMM:
    """One model on one metadata table; fits every channel of a (rows, channels) matrix."""

    def __init__(self, meta: pd.DataFrame, spec: LMMSpec):
        self.spec = spec
        self.meta = meta.reset_index(drop=True)
        self.grouping = spec.grouping
        self.terms = _terms(spec.fixed, [c for c in self.meta.columns if c != "POWER"])
        self.vars = sorted({v for t in self.terms for v in t} - {"POWER"})
        cats = set(spec.categorical) if spec.categorical is not None else {
            spec.subject, spec.group, self.grouping}
        need = set(self.vars) | {self.grouping} | ({spec.dv} - {"POWER"})
        for c in sorted(need):
            if c not in self.meta.columns:
                raise ValueError(f"Model column '{c}' is not in the metadata")
            bad = np.flatnonzero(_is_missing(self.meta[c]))
            if bad.size:   # the pipeline drops these rows first (drop_missing_meta)
                raise ValueError(
                    f"Model column '{c}' has a missing value in row(s) "
                    f"{', '.join(str(i + 1) for i in bad[:10])}{' ...' if bad.size > 10 else ''}; "
                    "drop those rows first (drop_missing_meta).")
        self.categorical = {v for v in self.vars
                            if v in cats or not pd.api.types.is_numeric_dtype(self.meta[v])}
        self.levels = {v: _sorted_levels(self.meta[v]) for v in self.categorical}
        self.power_is_dv = spec.dv == "POWER"
        if self.power_is_dv and any("POWER" in t for t in self.terms):
            raise ValueError("POWER cannot be both the response and a fixed effect")
        if not self.power_is_dv and not any("POWER" in t for t in self.terms):
            raise ValueError("the fixed part must reference POWER unless POWER is the response")
        codes, _ = pd.factorize(self.meta[self.grouping], sort=False)
        self.gidx = codes
        self.n_groups = int(codes.max()) + 1
        self.names, self.term_cols = self._names()
        self.effect_cols = self._effect_cols()

    # ---- design
    def _blocks(self, v, meta, power):
        """Columns and names of one variable: (n, k) for a factor (reference coded), (n, 1) else."""
        if v == "POWER":
            return [power], ["POWER"]
        if v in self.categorical:
            lev = self.levels[v]
            col = meta[v].to_numpy()
            return ([(col == lv).astype(float) for lv in lev[1:]],
                     [f"{v}_{lv}" for lv in lev[1:]])
        return [meta[v].to_numpy(dtype=float)], [v]

    def _names(self):
        names, term_cols = ["(Intercept)"], []
        dummy_power = np.zeros(len(self.meta))
        for t in self.terms:
            cols = [np.ones(len(self.meta))]
            nm = [""]
            for v in t:
                c, n = self._blocks(v, self.meta, dummy_power)
                cols = [a * b for a in cols for b in c]
                nm = [f"{a}:{b}" if a else b for a in nm for b in n]
            term_cols.append(list(range(len(names), len(names) + len(nm))))
            names += nm
        return names, term_cols

    def _effect_cols(self) -> list[int]:
        e = self.spec.effect
        if self.spec.effect_type == "factor":
            want = set(e.split(":"))
            for t, cols in zip(self.terms, self.term_cols):
                if set(t) == want:
                    return cols
            raise ValueError(f"Effect '{e}' is not a term of the fixed part '{self.spec.fixed}'")
        if e in self.names:
            return [self.names.index(e)]
        hit = [i for i, n in enumerate(self.names) if e in n]   # MATLAB: contains(names, effect)
        if not hit:
            raise ValueError(f"Effect '{e}' is not a coefficient of the fixed part "
                             f"'{self.spec.fixed}' (coefficients: {', '.join(self.names)})")
        return hit[:1]

    def design(self, power: np.ndarray, meta: pd.DataFrame | None = None):
        """``(X, y)``: X is (channels, rows, p); y is (rows,) or (rows, channels)."""
        meta = self.meta if meta is None else meta
        power = np.asarray(power, dtype=float)
        n, nch = power.shape
        const = []
        for t in self.terms:
            if "POWER" in t:
                continue
            cols = [np.ones(n)]
            for v in t:
                c, _ = self._blocks(v, meta, None)
                cols = [a * b for a in cols for b in c]
            const.append(cols)
        X = np.empty((nch, n, len(self.names)))
        X[:, :, 0] = 1.0
        ci = iter(const)
        for t, cols_idx in zip(self.terms, self.term_cols):
            if "POWER" in t:
                rest = [v for v in t if v != "POWER"]
                cols = [np.ones(n)]
                for v in rest:
                    c, _ = self._blocks(v, meta, None)
                    cols = [a * b for a in cols for b in c]
                for k, c in zip(cols_idx, cols):
                    X[:, :, k] = (power * c[:, None]).T
            else:
                for k, c in zip(cols_idx, next(ci)):
                    X[:, :, k] = c[None, :]
        y = power if self.power_is_dv else meta[self.spec.dv].to_numpy(dtype=float)
        return X, y

    # ---- fit
    def fit(self, power, meta: pd.DataFrame | None = None, evaluable=None) -> FitResult:
        """Fit every evaluable channel. Channels outside ``evaluable`` are not fitted."""
        power = np.asarray(power, dtype=float)
        nch = power.shape[1]
        evaluable = np.ones(nch, bool) if evaluable is None else np.asarray(evaluable, bool)
        if evaluable.size != nch:
            raise ValueError(f"evaluable mask has {evaluable.size} entries but power has "
                             f"{nch} channels")
        out = FitResult.empty(nch, len(self.names))
        if power.shape[0] < MIN_TRIALS or not evaluable.any():
            return out
        idx = np.flatnonzero(evaluable)
        X, y = self.design(power[:, idx], meta)
        fit = fit_random_intercept(y, X, self.gidx, self.n_groups)
        out.fill(idx, fit, self.effect_cols, self.spec.effect_type)
        return out


@dataclass
class FitResult:
    """Per-channel output; NaN (Wald 0) where no fit was made or the statistic is not finite."""

    stat: np.ndarray
    wald: np.ndarray
    p_model: np.ndarray
    beta: np.ndarray        # (channels, p)
    se: np.ndarray          # (channels, p)
    df: int | float
    loglik: np.ndarray
    sigma: np.ndarray       # residual SD
    sd_subject: np.ndarray  # random-intercept SD
    theta: np.ndarray       # sd_subject / sigma
    boundary: np.ndarray    # subject variance estimated as exactly zero
    degenerate: np.ndarray = None   # no residual variation left (degenerate()): no statistic

    @classmethod
    def empty(cls, nch, p):
        nan = np.full(nch, np.nan)
        return cls(nan.copy(), np.zeros(nch), nan.copy(), np.full((nch, p), np.nan),
                   np.full((nch, p), np.nan), np.nan, nan.copy(), nan.copy(), nan.copy(),
                   nan.copy(), np.zeros(nch, bool), np.zeros(nch, bool))

    def fill(self, idx, fit, effect_cols, effect_type):
        self.beta[idx], self.se[idx] = fit["beta"], fit["se"]
        self.df = fit["df"]
        for k in ("loglik", "theta"):
            getattr(self, k)[idx] = fit[k]
        self.sigma[idx] = np.sqrt(fit["sigma2"])
        self.sd_subject[idx] = fit["theta"] * np.sqrt(fit["sigma2"])
        self.boundary[idx] = fit["boundary"]
        self.degenerate[idx] = fit["degenerate"]
        e = list(effect_cols)
        with np.errstate(all="ignore"):
            if effect_type == "continuous":
                s = fit["beta"][:, e[0]] / fit["se"][:, e[0]]
                p = 2 * stats.t.sf(np.abs(s), fit["df"])
                w = s ** 2
            else:
                b = fit["beta"][:, e]
                C = fit["cov"][:, e][:, :, e]
                s = np.full(len(idx), np.nan)
                ok = np.isfinite(C).all(axis=(1, 2)) & np.isfinite(b).all(axis=1)
                if ok.any():   # a zero or non-positive-definite covariance has no F
                    ok[ok] = np.linalg.eigvalsh(C[ok]).min(axis=1) > 0
                if ok.any():
                    s[ok] = np.einsum("cp,cp->c", b[ok], np.linalg.solve(
                        C[ok], b[ok][..., None])[..., 0]) / len(e)
                p = stats.f.sf(s, len(e), fit["df"])
                w = s
        good = np.isfinite(s)
        self.stat[idx[good]] = s[good]
        self.wald[idx[good]] = w[good]
        self.p_model[idx[good]] = p[good]


# --------------------------------------------------------------------------- the estimator

def _sufficient(y, X, gidx, n_groups):
    nch, n, _ = X.shape
    Y = np.broadcast_to(y[:, None], (n, nch)) if y.ndim == 1 else y
    G = np.zeros((n_groups, n))
    G[gidx, np.arange(n)] = 1.0
    return {
        "n_i": G.sum(axis=1), "N": n,
        "Sx": np.einsum("gn,cnp->cgp", G, X), "Sy": (G @ Y).T,
        "XtX": np.einsum("cnp,cnq->cpq", X, X), "Xty": np.einsum("cnp,nc->cp", X, Y),
        "yty": np.einsum("nc,nc->c", Y, Y)}


def _solve(A, b):
    """Batched A x = b; a channel whose A is singular gets NaN instead of stopping the run."""
    try:
        return np.linalg.solve(A, b[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.full(b.shape, np.nan)
        for c in range(A.shape[0]):
            try:
                out[c] = np.linalg.solve(A[c], b[c])
            except np.linalg.LinAlgError:
                pass
        return out


def _inv(A):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        out = np.full(A.shape, np.nan)
        for c in range(A.shape[0]):
            try:
                out[c] = np.linalg.inv(A[c])
            except np.linalg.LinAlgError:
                pass
        return out


def _gls(s, theta):
    """Generalised least squares at relative intercept variance theta**2, all channels."""
    g = theta ** 2
    w = g[:, None] / (1.0 + s["n_i"][None, :] * g[:, None])
    XtWX = s["XtX"] - np.einsum("cg,cgp,cgq->cpq", w, s["Sx"], s["Sx"])
    XtWy = s["Xty"] - np.einsum("cg,cgp,cg->cp", w, s["Sx"], s["Sy"])
    ytWy = s["yty"] - np.einsum("cg,cg->c", w, s["Sy"] ** 2)
    beta = _solve(XtWX, XtWy)
    r2 = ytWy - np.einsum("cp,cp->c", XtWy, beta)
    logdet = np.log1p(s["n_i"][None, :] * g[:, None]).sum(axis=1)
    return XtWX, beta, r2, logdet


def _deviance(s, theta):
    """-2 x profiled log-likelihood, up to a constant: N log r2(theta) + log det."""
    _, _, r2, logdet = _gls(s, theta)
    with np.errstate(divide="ignore", invalid="ignore"):
        return s["N"] * np.log(np.where(r2 > 0, r2, np.nan)) + logdet


def _polish(s, th, h=1e-5, steps=3):
    """Newton steps on central-difference derivatives. Function values fix theta to about 1e-7
    (the deviance is flat to rounding near its minimum); the derivative fixes it far better."""
    for _ in range(steps):
        hh = np.maximum(h, h * th)
        fp, f0, fm = _deviance(s, th + hh), _deviance(s, th), _deviance(s, np.abs(th - hh))
        grad, curv = (fp - fm) / (2 * hh), (fp - 2 * f0 + fm) / hh ** 2
        step = grad / np.where(curv > 0, curv, 1.0)
        # The deviance is too flat near the minimum to compare values, so a step is taken on the
        # derivative alone; it is kept only when small relative to theta (a local correction).
        ok = (np.isfinite(step) & np.isfinite(curv) & (curv > 0) & (th > hh)
              & (np.abs(step) <= 0.1 * th))
        th = np.where(ok, th - step, th)
    return th


def _rss_rank(A, Y):
    """Residual sum of squares of each column of Y on its channel's A, and rank(A), batched.

    A is (channels, rows, k), Y is (rows, channels). Rank uses MATLAB's tolerance
    max(size(A)) * eps(largest singular value); the projection is the minimum-norm one."""
    U, S, _ = np.linalg.svd(A, full_matrices=False)
    tol = max(A.shape[1], A.shape[2]) * np.spacing(S[:, :1])
    keep = S > tol
    proj = np.einsum("cnk,ck,nc->ck", U, keep, Y)
    fit = np.einsum("cnk,ck->nc", U, proj)
    return ((Y - fit) ** 2).sum(axis=0), keep.sum(axis=1)


def within_ratio(V, subj) -> np.ndarray:
    """Within-subject share of the variation of each column of V (``snpm_lmm_within_ratio.m``):
    SS_within / SS_total, grouped by subject only; 0 when SS_total <= 1e-20 sum(v^2) (a flat
    column); NaN for a column with a non-finite value."""
    V = np.asarray(V, dtype=float)
    V = V[:, None] if V.ndim == 1 else V
    codes, _ = pd.factorize(pd.Series(np.asarray(subj)), sort=False)
    q = int(codes.max()) + 1
    cnt = np.bincount(codes, minlength=q).astype(float)
    w = np.full(V.shape[1], np.nan)
    for j in range(V.shape[1]):
        v = V[:, j]
        if not np.isfinite(v).all():
            continue
        m = np.bincount(codes, weights=v, minlength=q) / cnt
        ssw = ((v - m[codes]) ** 2).sum()
        sst = ((v - v.mean()) ** 2).sum()
        w[j] = 0.0 if sst <= ZERO_SS_RTOL * (v ** 2).sum() else ssw / sst
    return w


def degenerate_info(y, X, gidx, n_groups) -> dict:
    """The quantities of ``snpm_lmm_degenerate.m`` per channel: rss_x, rss_xz, tol, theta2
    (method of moments; NaN when df_b or df_w is not positive) and the decision ``degenerate``."""
    X = np.asarray(X, dtype=float)
    nch, n, _ = X.shape
    Y = np.broadcast_to(np.asarray(y, dtype=float)[:, None], (n, nch)) if np.ndim(y) == 1 \
        else np.asarray(y, dtype=float)
    tol = ZERO_SS_RTOL * (Y ** 2).sum(axis=0)
    rss_x, rank_x = _rss_rank(X, Y)
    Z = np.zeros((n, n_groups))
    Z[np.arange(n), gidx] = 1.0
    XZ = np.concatenate([X, np.broadcast_to(Z, (nch, n, n_groups))], axis=2)
    rss_xz, rank_xz = _rss_rank(XZ, Y)
    df_b, df_w = rank_xz - rank_x, n - rank_xz
    ok = (df_b > 0) & (df_w > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        F_b = ((rss_x - rss_xz) / df_b) / (rss_xz / df_w)
        theta2 = np.where(ok, (F_b - 1) / (n / n_groups), np.nan)
    # rss_xz <= tol counts only when [X Z] does not saturate the rows (rank < N): with one trial
    # per subject, say, [X Z] fits any y exactly and says nothing about the data.
    exact = (rss_x <= tol) | ((rank_xz < n) & (rss_xz <= tol))
    theta2 = np.where(exact, np.nan, theta2)
    with np.errstate(invalid="ignore"):
        deg = exact | (ok & (theta2 >= THETA_MAX ** 2))
    return {"rss_x": rss_x, "rss_xz": rss_xz, "tol": tol, "theta2": theta2, "degenerate": deg,
            "rank_x": rank_x, "rank_xz": rank_xz}


def degenerate(y, X, gidx, n_groups) -> np.ndarray:
    """Channels with no residual variation left, so the t or F is undefined
    (``snpm_lmm_degenerate.m``; the zero-variance rule of the other tiers, for the LMM).

    Checked on the data, for each channel: RSS(y ~ X) <= 1e-20 y'y (e.g. a flat channel), or
    RSS(y ~ [X Z]) <= 1e-20 y'y with Z the subject indicators (e.g. a channel constant within each
    subject), or the method-of-moments subject/residual SD ratio
    theta^2 = ((RSS_X - RSS_XZ)/df_b / (RSS_XZ/df_w) - 1) / (N/q) at or above THETA_MAX^2
    (THETA_MAX = 1e3). The decision is made with this moment ratio, never with the ML theta of
    the fit, which stays the source of the statistic.
    """
    return degenerate_info(y, X, gidx, n_groups)["degenerate"]


def fit_random_intercept(y, X, gidx, n_groups, xtol=1e-10) -> dict:
    """Maximum-likelihood fit of ``y ~ X + (1|g)`` for every channel at once.

    ``X`` is (channels, rows, p); ``y`` is (rows,) or (rows, channels). Channels whose design is
    rank deficient, or that are :func:`degenerate`, come back NaN. Returns beta, se, cov
    (conditional on theta, as fitlme), df, sigma2, theta, loglik and the boundary flag, each per
    channel.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    nch, n, p = X.shape
    gidx = np.asarray(gidx)
    full = np.linalg.matrix_rank(X) == p
    deg = np.zeros(nch, bool)
    if full.any():   # no residual variation left: no statistic (fitted rows only)
        bad = degenerate(y if y.ndim == 1 else y[:, full], X[full], gidx, n_groups)
        deg[np.flatnonzero(full)[bad]] = True
        full[deg] = False
    out = {"beta": np.full((nch, p), np.nan), "se": np.full((nch, p), np.nan),
               "cov": np.full((nch, p, p), np.nan), "df": n - p, "sigma2": np.full(nch, np.nan),
               "theta": np.full(nch, np.nan), "loglik": np.full(nch, np.nan),
               "boundary": np.zeros(nch, bool), "degenerate": deg}
    if not full.any():
        return out
    yk = y if y.ndim == 1 else y[:, full]
    s = _sufficient(yk, X[full], gidx, n_groups)
    m = int(full.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.stack([_deviance(s, np.full(m, t)) for t in _GRID])
        dev = np.where(np.isfinite(dev), dev, np.inf)
        k = dev.argmin(axis=0)
        top = k == _GRID.size - 1
        if top.any():   # subject SD over 1e4 residual SDs: widen the bracket upwards
            hi = np.logspace(4, 8, 33)
            dev_hi = np.stack([_deviance(s, np.full(m, t)) for t in hi])
            k_hi = np.nanargmin(np.where(np.isfinite(dev_hi), dev_hi, np.inf), axis=0)
        a = _GRID[np.maximum(k - 1, 0)]
        b = _GRID[np.minimum(k + 1, _GRID.size - 1)]
        if top.any():
            a = np.where(top, np.r_[_GRID[-2], hi][np.maximum(k_hi, 0)], a)
            b = np.where(top, hi[np.minimum(k_hi + 1, hi.size - 1)], b)
        for _ in range(300):
            if np.max(b - a) <= xtol * np.maximum(1.0, np.max(b)):
                break
            c = b - _GOLD * (b - a)
            d = a + _GOLD * (b - a)
            left = _deviance(s, c) < _deviance(s, d)
            b = np.where(left, d, b)
            a = np.where(left, a, c)
        th = _polish(s, (a + b) / 2)
        # Zero subject variance is the optimum when the deviance rises as the variance g = theta^2
        # leaves zero. Its slope in g is measured over g = 1e-8, where the rise (about N x 1e-8)
        # is far above rounding; comparing values at the optimum itself would not be. A search
        # that ended away from zero (theta >= 1e-3) found a lower interior minimum and stands.
        d0 = _deviance(s, np.zeros(m))
        rises = _deviance(s, np.full(m, 1e-4)) >= d0
        at_zero = (rises & (th < 1e-3)) | (d0 <= _deviance(s, th))
        th = np.where(at_zero, 0.0, th)
        XtWX, beta, r2, logdet = _gls(s, th)
        sigma2 = r2 / n
        cov = sigma2[:, None, None] * _inv(XtWX)
        se = np.sqrt(np.einsum("cpp->cp", cov))
        loglik = -0.5 * (n * np.log(2 * np.pi * sigma2) + n + logdet)
    # Degenerate channels were removed before the fit (degenerate()); anything that still has
    # no finite variance here (a numerically singular system) gets no statistic either.
    fitted = np.isfinite(sigma2) & (sigma2 > 0)
    full_idx = np.flatnonzero(full)
    keep, lose = full_idx[fitted], full_idx[~fitted]
    out["beta"][keep], out["se"][keep], out["cov"][keep] = beta[fitted], se[fitted], cov[fitted]
    out["sigma2"][keep], out["theta"][keep] = sigma2[fitted], th[fitted]
    out["loglik"][keep] = loglik[fitted]
    out["boundary"][keep] = at_zero[fitted]
    out["sigma2"][lose] = 0.0
    return out


# --------------------------------------------------------------------------- permutations

def _subject_codes(meta, subject):
    codes, _ = pd.factorize(meta[subject], sort=False)   # unique(..., 'stable')
    return codes


def draw_perms(meta: pd.DataFrame, spec: LMMSpec, n_perm: int,
               rng: np.random.Generator) -> np.ndarray:
    """Permutation indices for ``spec.perm`` (0-based, one row per permutation).

    ``within_subject``: (n_perm, rows) row indices, shuffled inside each subject's rows; the
    response of row r becomes the response of row ``perm[r]``. ``group_label``: (n_perm,
    subjects) indices into the subjects in order of first appearance; subject s receives the
    group of subject ``perm[s]``. As ``snpm_lmm_permute_meta.m``, which records these vectors.
    """
    codes = _subject_codes(meta, spec.subject)
    if spec.perm == "within_subject":
        return _perm.freedman_lane_indices(len(meta), n_perm, "within", codes, rng)
    return _perm.freedman_lane_indices(int(codes.max()) + 1, n_perm, "free", None, rng)


def permute_meta(meta: pd.DataFrame, spec: LMMSpec, perm_row) -> pd.DataFrame:
    """One permuted copy of the metadata (``snpm_lmm_permute_meta.m``)."""
    perm_row = np.asarray(perm_row, dtype=int)
    if np.array_equal(perm_row, np.arange(perm_row.size)):
        return meta                     # the identity: the observed labelling, unchanged
    out = meta.copy()
    if spec.perm == "within_subject":
        if spec.dv not in meta.columns:
            raise ValueError("within_subject permutation shuffles the response column, but the "
                             f"response '{spec.dv}' is not a metadata column (use group_label)")
        out[spec.dv] = meta[spec.dv].to_numpy()[perm_row]
    else:
        codes = _subject_codes(meta, spec.subject)
        first = np.array([np.flatnonzero(codes == s)[0] for s in range(codes.max() + 1)])
        subj_groups = meta[spec.group].to_numpy()[first]
        out[spec.group] = subj_groups[perm_row[codes]]
    return out


def _resolve(meta, spec, n_perm, rng, perms):
    if perms is not None:
        return np.asarray(perms, dtype=int)
    if n_perm is None:
        raise ValueError("give n_perm or perms")
    return draw_perms(meta, spec, int(n_perm), rng if rng is not None else np.random.default_rng())


def _evaluable(power, evaluable):
    power = np.asarray(power, dtype=float)
    ev = (np.isfinite(power).all(axis=0) if evaluable is None
          else np.asarray(evaluable, bool).ravel())
    if ev.size != power.shape[1]:
        raise ValueError(f"evaluable mask has {ev.size} entries but power has "
                         f"{power.shape[1]} channels")
    if not ev.any():
        raise ValueError(f"No channel is complete across all {power.shape[0]} analysed rows, so "
                         "no channel can be tested against a permutation null. Drop the "
                         "incomplete subjects/trials instead of the channels.")
    return ev


# --------------------------------------------------------------------------- TFCE

@dataclass
class LMMTFCEResult:
    """Output of :func:`lmm_tfce` (field names follow MATLAB's ``T`` and ``p``)."""

    real_T: np.ndarray
    real_TFCE: np.ndarray
    tMaxTFCE: np.ndarray          # sorted descending
    critical_T_TFCE: float
    p_real: np.ndarray            # model p (two-sided t, or F)
    p_correctedTFCE: np.ndarray   # NaN outside the evaluable set
    excluded_channels: np.ndarray
    n_degenerate: int             # permutations with no finite statistic (entered as +Inf)
    n_perm: int
    fit: FitResult = field(repr=False)
    perms: np.ndarray = field(repr=False)
    # observed channels caught by the per-labelling backstop (degenerate()): NaN in the observed
    # map, still in every permuted map, listed as PERFECT_FIT (MATLAB T.undefined_channels)
    undefined_channels: np.ndarray = field(default_factory=lambda: np.empty(0, int))

    @property
    def excluded_reasons(self) -> list[str]:
        return [MISSING] * self.excluded_channels.size


def _null_max(enhanced, evaluable, tail):
    """Largest enhanced value over the evaluable channels; +Inf when there is none."""
    cand = enhanced[evaluable]
    if tail == "both":
        cand = np.abs(cand)
    cand = cand[~np.isnan(cand)]
    if cand.size == 0:
        return np.inf
    mx = cand.max()
    return mx if np.isfinite(mx) else np.inf


def _enhance(stat, graph, E, H):
    if not np.isfinite(stat).any():
        return np.full(stat.shape, np.nan)
    return tfce(stat, graph, E, H)


def tfce_correction(real_stat, real_p, perm_stats, graph: ChannelGraph, evaluable,
                    tail: str = "both", alpha: float = 0.05, E: float = 0.5,
                    H: float = 2.0) -> dict:
    """TFCE max-statistic correction from an observed map and an iterable of permuted maps.

    The second half of ``snpm_lmm_TFCE.m``, separated from the refits so that the correction can
    be checked on MATLAB's recorded permuted maps. Every map is restricted to ``evaluable``.
    """
    if tail not in ("both", "left", "right"):
        raise ValueError("tail must be 'both', 'left' or 'right'")
    ev = np.asarray(evaluable, bool)
    real_T = np.where(ev, np.asarray(real_stat, dtype=float), np.nan)
    p_real = np.where(ev, np.asarray(real_p, dtype=float), np.nan)
    real_TFCE = _enhance(real_T, graph, E, H)
    real_TFCE[np.isnan(real_T)] = np.nan
    null = []
    for s in perm_stats:
        s = np.where(ev, np.asarray(s, dtype=float), np.nan)
        null.append(_null_max(_enhance(s, graph, E, H), ev, tail))
    max_tfce = np.asarray(null, dtype=float)
    n = max_tfce.size
    n_deg = int(np.isinf(max_tfce).sum())
    if n_deg:
        warnings.warn(f"{n_deg} of {n} permutations produced no finite statistic on any "
                      "evaluable channel; those null entries were set to +Inf (conservative) "
                      "rather than dropped. Check the model.", RuntimeWarning, stacklevel=3)
    max_tfce = _sort_desc(max_tfce)
    p_tfce = _fwe_p(real_TFCE, max_tfce, two_sided=tail == "both")
    p_tfce[~ev] = np.nan
    return {"real_T": real_T, "real_TFCE": real_TFCE, "tMaxTFCE": max_tfce,
                "critical_T_TFCE": max_tfce[int(np.floor(alpha * n))], "p_real": p_real,
                "p_correctedTFCE": p_tfce, "excluded_channels": np.flatnonzero(~ev),
                "n_degenerate": n_deg, "n_perm": n}


def _setup(power, meta, spec, graph, evaluable, n_perm, rng, perms):
    power = np.asarray(power, dtype=float)
    if power.shape[1] != graph.n:
        raise ValueError(f"power has {power.shape[1]} channels but the channel graph has {graph.n}")
    ev = _evaluable(power, evaluable)
    model = LMM(meta, spec)
    return power, ev, model, _resolve(model.meta, spec, n_perm, rng, perms)


def lmm_tfce(power, meta: pd.DataFrame, spec: LMMSpec, graph: ChannelGraph,
             tail: str = "both", alpha: float = 0.05, n_perm: int | None = None,
             E: float = 0.5, H: float = 2.0, evaluable=None,
             rng: np.random.Generator | None = None, perms=None) -> LMMTFCEResult:
    """Per-channel LMM statistic with TFCE family-wise correction (``snpm_lmm_TFCE.m``)."""
    check_design(spec, tail, list(meta.columns))
    power, ev, model, perms = _setup(power, meta, spec, graph, evaluable, n_perm, rng, perms)
    real = model.fit(power, evaluable=ev)
    maps = (model.fit(power, permute_meta(model.meta, spec, row), ev).stat for row in perms)
    out = tfce_correction(real.stat, real.p_model, maps, graph, ev, tail, alpha, E, H)
    res = LMMTFCEResult(**out, fit=real, perms=perms)
    res.undefined_channels = np.flatnonzero(real.degenerate & ev)
    return res


# --------------------------------------------------------------------------- cluster mass

@dataclass
class LMMCluster:
    channels: np.ndarray   # 0-based; empty when no channel passed the threshold
    mass: float            # mean Wald over the cluster: sum(Wald) / number of channels
    p: float
    threshold: float
    permutations: int


@dataclass
class LMMClusterResult:
    clusters: list[LMMCluster]
    maxMass: np.ndarray    # sorted descending
    n_perm: int
    perms: np.ndarray = field(repr=False)


def _clusters(p_model, wald, alpha, graph):
    """Connected channels with model p < alpha, each with mass sum(Wald)/n_channels."""
    with np.errstate(invalid="ignore"):
        mask = np.where(np.isnan(p_model), 0.0, (p_model < alpha).astype(float))
    return [(c, float(np.nansum(wald[c])) / c.size) for c in find_clusters(mask, 0.5, graph)]


def cluster_correction(real_p, real_wald, perm_maps, graph: ChannelGraph, evaluable,
                       alpha: float = 0.05) -> tuple[list[LMMCluster], np.ndarray]:
    """Cluster-mass correction from the observed (p, Wald) maps and an iterable of permuted
    (p, Wald) maps: the second half of ``snpm_lmm_cluster.m``. Outside ``evaluable`` a channel
    has no p and no mass in every map."""
    ev = np.asarray(evaluable, bool)

    def masked(p, w):
        return (np.where(ev, np.asarray(p, dtype=float), np.nan),
                np.where(ev, np.asarray(w, dtype=float), 0.0))

    real = _clusters(*masked(real_p, real_wald), alpha, graph)
    null = []
    for p, w in perm_maps:
        found = _clusters(*masked(p, w), alpha, graph)
        null.append(max(m for _, m in found) if found else 0.0)
    max_mass = _sort_desc(np.asarray(null, dtype=float))
    n = max_mass.size
    if not real:
        return [LMMCluster(np.empty(0, dtype=int), 0.0, 1.0, alpha, n)], max_mass
    return [LMMCluster(c, m, (int(_at_least(max_mass, m).sum()) + 1) / (n + 1), alpha, n)
            for c, m in real], max_mass


def lmm_cluster(power, meta: pd.DataFrame, spec: LMMSpec, graph: ChannelGraph,
                alpha: float = 0.05, n_perm: int | None = None, evaluable=None,
                rng: np.random.Generator | None = None, perms=None) -> LMMClusterResult:
    """Cluster-mass permutation test (``snpm_lmm_cluster.m``): clusters of channels with model
    p < alpha (two-sided for a continuous effect), mass = mean Wald (t^2 or F), null of the
    largest mass per permutation (0 when a permutation has no cluster)."""
    check_design(spec, "both", list(meta.columns))
    power, ev, model, perms = _setup(power, meta, spec, graph, evaluable, n_perm, rng, perms)
    real = model.fit(power, evaluable=ev)

    def maps():
        for row in perms:
            f = model.fit(power, permute_meta(model.meta, spec, row), ev)
            yield f.p_model, f.wald

    clusters, max_mass = cluster_correction(real.p_model, real.wald, maps(), graph, ev, alpha)
    return LMMClusterResult(clusters, max_mass, perms.shape[0], perms)
