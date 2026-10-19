"""From input files to corrected maps for the t-test and correlation analyses.

Ports the flow of ``core_snpm_analysis.m``, in its order: read, match subjects (correlation),
check the two files line up, select channels, transform, drop empty rows, apply the channel rule
once, residualise on covariates, whole-head test, then the two permutation engines. ``run``
routes the GLM presets and the circular tier (``circ_pipeline``) as MATLAB's early guards do,
and the Tier-1 unsigned-measure guard runs before every non-circular analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy import linalg, stats

from . import circ_pipeline, legacy, missing
from .io import tables
from .montage import ChannelSetup, get_montage, setup_channels

ALPHA, TFCE_E, TFCE_H = 0.05, 0.5, 2.0
DEFAULT_PERMUTATIONS = 5000

FL_METHOD = ("Freedman-Lane, approximate: x and y residualized on [1, covariates]; each "
             "permutation re-projects the permuted EEG residuals off the covariates before "
             "correlating. Not an exact test.")
SPEARMAN_CAVEAT = ("Partial Spearman controls the covariates only as far as their relation to "
                   "the RANKS is linear (data and covariates are ranked, then the covariates are "
                   "partialled out linearly). A covariate whose effect is linear on the raw "
                   "scale is not fully removed. Use correlationP or the regression preset when "
                   "covariate control matters.")


class CovariateNoErrorDF(ValueError):
    """A partial correlation with no error degrees of freedom (n - 2 - k_eff < 1);
    MATLAB ``snpm:covariateNoErrorDF``."""


def _conservative_note(df: int, k_eff: int, n: int) -> str:
    """Low residual-df caveat (n - 2 - k_eff < 5), the MATLAB text verbatim."""
    return (f"Only {df} error degrees of freedom remain after partialling out {k_eff} "
            f"covariate(s) from {n} subjects (n - 2 - covariates < 5). With covariates the "
            "permutation test is APPROXIMATE (Freedman-Lane). At residual df below 5 the "
            "per-channel max-statistic family-wise error may be mildly liberal (measured 0.056 "
            "at nominal 0.05, df 3), while TFCE and cluster inference are conservative. "
            "Interpret borderline results cautiously.")


@dataclass
class CovariateInfo:
    """How the covariates were controlled (MATLAB ``results_struct.covariates``)."""

    names: tuple[str, ...]
    n_covariates: int            # columns supplied
    k_eff: int                   # rank([1, covariates]) - 1, used in every df statement
    error_df: int                # n - 2 - k_eff
    method: str = FL_METHOD
    spearman_caveat: str = ""
    conservative_note: str = ""

    def as_dict(self) -> dict:
        return {"names": list(self.names), "n_covariates": self.n_covariates,
                "k_eff": self.k_eff, "error_df": self.error_df, "method": self.method,
                "spearman_caveat": self.spearman_caveat,
                "conservative_note": self.conservative_note}


@dataclass
class Prepared:
    """Everything decided before the first permutation."""

    comparison: str              # what the user asked for
    stat_comparison: str         # what the engines are given (rank partial correlation -> Pearson)
    tail: str
    x: np.ndarray
    y: np.ndarray
    setup: ChannelSetup
    rule: missing.ChannelRule
    n_analysed: np.ndarray
    threshold: float             # cluster-forming threshold
    ncov: int = 0                        # k_eff: covariate dimensions partialled out
    covariate_names: tuple[str, ...] = ()
    subjects: tuple[str, ...] = ()       # analysed subjects (correlation only)
    behavior_col: str = ""
    notes: list[str] = field(default_factory=list)
    global_interpretable: bool = True
    undefined: np.ndarray | None = None  # t-tests: observed statistic undefined (still permuted)
    fl: legacy.FLContext | None = None   # partial correlation: Freedman-Lane context
    covariates: CovariateInfo | None = None

    @property
    def excluded(self) -> tuple[np.ndarray, list[str]]:
        """Every channel without an observed statistic, with its reason, in channel order."""
        why = {int(i): r for i, r in zip(self.rule.excluded, self.rule.reasons)}
        if self.undefined is not None:
            why.update({int(i): missing.PERFECT_SEPARATION for i in np.flatnonzero(self.undefined)})
        idx = sorted(why)
        return np.array(idx, dtype=int), [why[i] for i in idx]


@dataclass
class LegacyResults:
    prepared: Prepared
    T: legacy.MaxStatResult
    clusters: legacy.ClusterResult
    global_: legacy.GlobalResult
    alpha: float

    @property
    def uncorrsigch(self):
        with np.errstate(invalid="ignore"):
            return np.flatnonzero(self.T.p_real <= self.alpha)

    @property
    def correctTFCEsigch(self):
        with np.errstate(invalid="ignore"):
            return np.flatnonzero(self.T.p_correctedTFCE <= self.alpha)

    @property
    def SnPMsigch(self):
        sig = [c.channels for c in self.clusters.clusters if c.p <= self.alpha]
        return np.concatenate(sig).astype(int) if sig else np.empty(0, dtype=int)


def cluster_threshold(comparison: str, n1: int, n2: int, alpha: float = ALPHA) -> float:
    """t-tests: the two-sided critical t. Correlation: a fixed r that loosens with sample size."""
    if comparison in missing.T_TESTS:
        return float(stats.t.ppf(1 - alpha / 2, min(n1 - 1, n2 - 1)))
    n = min(n1, n2)
    return 0.50 if n <= 15 else 0.40 if n <= 30 else 0.35 if n <= 50 else 0.30


def _note(prep_notes: list[str], message: str):
    prep_notes.append(message)
    warnings.warn(message, stacklevel=3)


def _check_alignment(kind1, kind2, comparison, system):
    """Data 2 is read position by position against Data 1, so their columns must line up."""
    if kind2.kind == "behavior":
        if comparison not in missing.CORRELATIONS:
            raise ValueError(
                f"Data 2 is a single measure column ('{kind2.col_name}') plus a subject column, "
                f"but {comparison} compares two channel maps. One value per subject can only be "
                "used in a correlation.")
        return "behavior"
    n1c, n2c = len(kind1.channel_cols), len(kind2.channel_cols)
    if kind2.kind == "montage":
        if n1c > 0 and n1c != n2c:
            raise ValueError(f"Data 1 has {n1c} channel columns and Data 2 has {n2c}. The files "
                             "are paired channel by channel and must carry the same channels.")
        if n1c > 0 and n2c > 0:
            labels = set(get_montage(system).labels)
            c1, c2 = kind1.data_cols, kind2.data_cols
            bad = [(i, c) for i, c in enumerate(c1) if c in labels
                   and (i >= len(c2) or c2[i].lower() != c.lower())]
            if bad:
                shown = "; ".join(f"position {i + 1}: Data 1 '{c}' vs Data 2 "
                                  f"'{c2[i] if i < len(c2) else '<no column>'}'" for i, c in bad[:5])
                cause = (f" Data 1 also carries non-channel column(s) "
                         f"({', '.join(kind1.nonchannel_cols[:5])}), which shift every channel "
                         "after them; remove them." if kind1.nonchannel_cols else "")
                raise ValueError(
                    f"Data 1 and Data 2 do not list the same channels in the same order: "
                    f"{len(bad)} positions disagree ({shown}).{cause} The files are read position "
                    "by position, so channel i of Data 1 would be compared with a different "
                    "channel of Data 2.")
        return "label"
    n1d, n2d = len(kind1.data_cols), len(kind2.data_cols)
    if n1d > 0 and n1d != n2d:
        raise ValueError(f"Data 1 has {n1d} data columns and Data 2 has {n2d}, and Data 2's "
                         "columns are not recognised channel labels, so they cannot be matched.")
    return "positional"


def _drop_subject(table):
    col = tables.subject_column(table.columns)
    return table if col is None else table.drop(columns=col)


def _transform(a, datatype):
    if datatype == "absolute":
        return a
    if datatype == "logscale":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log10(a)
    if datatype == "normalize":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (a - np.nanmean(a, axis=1, keepdims=True)) / np.nanstd(a, axis=1, ddof=1,
                                                                         keepdims=True)
    raise ValueError(f"datatype must be 'absolute', 'logscale' or 'normalize', got {datatype!r}")


def _residual_forming(Z) -> tuple[np.ndarray, int]:
    """(Rz, k_eff) of the nuisance design D = [1, Z]: Rz = I - QQ' from a column-pivoted economy
    QR, rank by MATLAB's tolerance max(size(D)) * eps(max |diag R|), k_eff = rank - 1. A
    duplicated or dependent covariate then costs nothing and is counted correctly."""
    n = Z.shape[0]
    D = np.column_stack([np.ones(n), Z])
    Q, R, _ = linalg.qr(D, mode="economic", pivoting=True)
    rdiag = np.abs(np.diag(R))
    rank = int((rdiag > max(D.shape) * np.spacing(rdiag.max())).sum())
    Q = Q[:, :rank]
    return np.eye(n) - Q @ Q.T, rank - 1


def _residualise(a, Rz):
    """Rz a for every complete column; a column with a missing value stays NaN."""
    out = np.full(a.shape, np.nan)
    ok = np.isfinite(a).all(axis=0)
    out[:, ok] = Rz @ a[:, ok]
    return out


def prepare(params: dict) -> Prepared:
    """Read the files named in ``params`` and return the matrices the engines will be given.

    Keys follow the MATLAB toolbox: ``data1_file``, ``data2_file``, ``data1_sheet``,
    ``data2_sheet``, ``comparison``, ``tail``, ``channels``, ``datatype``, ``use_covariates``,
    ``covariate_file``.
    """
    comparison = params["comparison"]
    if comparison not in legacy.COMPARISONS:
        raise ValueError(f"unsupported comparison {comparison!r}")
    circ_pipeline.tier1_guard(params)
    tail = params.get("tail", "both")
    datatype = params.get("datatype", "absolute")
    system = params.get("channels", "egi")
    is_corr = comparison in missing.CORRELATIONS
    notes: list[str] = []

    t1 = tables.read_table(params["data1_file"], params.get("data1_sheet"))
    vs_zero = comparison == "onesampleT" and not params.get("data2_file")
    t2 = t1 if vs_zero else tables.read_table(params["data2_file"], params.get("data2_sheet"))

    if is_corr and (tables.subject_column(t1.columns) is None
                    or tables.subject_column(t2.columns) is None):
        raise ValueError("Correlation needs a subject column in both files.")

    mode = ""
    if not vs_zero:
        kind2 = tables.detect_table(t2.columns)
        mode = _check_alignment(tables.detect_table(t1.columns), kind2, comparison, system)
        if mode == "positional":
            _note(notes, "Data 2's columns are not recognised channel labels, so Data 2 is matched "
                         "to Data 1 by position; the pairing cannot be checked by name.")
    behavior_col = kind2.col_name if mode == "behavior" else ""

    subjects: tuple[str, ...] = ()
    cov = None
    cov_names: tuple[str, ...] = ()
    if is_corr:
        s1 = tables.subject_ids(t1[tables.subject_column(t1.columns)])
        s2 = tables.subject_ids(t2[tables.subject_column(t2.columns)])
        pos2 = {}
        for j, s in enumerate(s2):
            pos2.setdefault(s, j)
        seen, i1, i2 = set(), [], []
        for i, s in enumerate(s1):
            if s in pos2 and s not in seen:
                seen.add(s)
                i1.append(i)
                i2.append(pos2[s])
        common = [s1[i] for i in i1]
        if len(common) < len(s1) or len(common) < len(s2):
            _note(notes, f"Subject matching: Data 1 has {len(s1)} subjects, Data 2 has "
                         f"{len(s2)}, {len(common)} are in both and are analysed.")
        t1, t2 = t1.iloc[i1].reset_index(drop=True), t2.iloc[i2].reset_index(drop=True)
        if behavior_col:
            score = t2[behavior_col]
            if not tables.pd.api.types.is_numeric_dtype(score):
                raise ValueError(f"The behavioural column '{behavior_col}' is not numeric; "
                                 "export text cells such as 'NA' as empty.")
            bad = ~np.isfinite(score.to_numpy(dtype=float))
            if bad.any():
                _note(notes, f"{int(bad.sum())} matched subject(s) have no value for "
                             f"'{behavior_col}' and are dropped: "
                             f"{', '.join(np.array(common, dtype=object)[bad])}.")
                t1, t2 = t1[~bad].reset_index(drop=True), t2[~bad].reset_index(drop=True)
                common = [s for s, b in zip(common, bad) if not b]
            if len(common) < 3:
                raise ValueError(f"Only {len(common)} subjects have both EEG data and a value "
                                 f"for '{behavior_col}'; a correlation needs at least 3.")
            if t2[behavior_col].nunique() < 2:
                raise ValueError(f"The behavioural measure '{behavior_col}' is constant across "
                                 "the analysed subjects.")
        if len(common) < 3:
            raise ValueError(f"Correlation needs at least 3 matching subjects; found {len(common)}.")
        if params.get("use_covariates") and params.get("covariate_file"):
            ct = tables.read_table(params["covariate_file"])
            cs = list(tables.subject_ids(ct.iloc[:, 0]))
            cov_names = tuple(str(c) for c in ct.columns[1:])
            keep = [i for i, s in enumerate(common) if s in cs]
            if len(keep) < len(common):
                _note(notes, f"{len(common) - len(keep)} subject(s) are not in the covariate "
                             "file and are excluded.")
                t1, t2 = t1.iloc[keep].reset_index(drop=True), t2.iloc[keep].reset_index(drop=True)
                common = [common[i] for i in keep]
            cov = ct.iloc[[cs.index(s) for s in common], 1:].to_numpy(dtype=float)
        subjects = tuple(common)

    d1 = _drop_subject(t1)
    x_all = tables.numeric_matrix(d1, "Data 1")
    labels = [str(c) for c in d1.columns]
    if behavior_col:
        setup = setup_channels(system, [x_all], labels)
        x = setup.data[0]
        y = np.repeat(t2[behavior_col].to_numpy(dtype=float)[:, None], x.shape[1], axis=1)
    else:
        y_all = tables.numeric_matrix(_drop_subject(t2), "Data 2")
        if y_all.shape[1] < x_all.shape[1]:
            raise ValueError("Data 2 has fewer data columns than Data 1.")
        setup = setup_channels(system, [x_all, y_all[:, :x_all.shape[1]]], labels)
        x, y = setup.data
    x = np.array(x, dtype=float)
    y = np.zeros_like(x) if vs_zero else np.array(y, dtype=float)

    x = _transform(x, datatype)
    if not behavior_col:
        y = _transform(y, datatype)

    # rows with no data at all
    ex, ey = np.isnan(x).all(axis=1), np.isnan(y).all(axis=1)
    if comparison != "unpairedT":
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"{comparison} pairs the files row by row, but Data 1 has "
                             f"{x.shape[0]} rows and Data 2 has {y.shape[0]}.")
        drop = ex | ey
        x, y = x[~drop], y[~drop]
        if cov is not None and cov.shape[0] == drop.size:
            cov = cov[~drop]
        subjects = tuple(s for s, d in zip(subjects, drop) if not d) if subjects else subjects
    else:
        x, y = x[~ex], y[~ey]
    if x.shape[0] < 2 or y.shape[0] < 2:
        raise ValueError(f"Too few subjects with data: {x.shape[0]} and {y.shape[0]}.")

    # MATLAB sets the threshold here, before a subject with a missing covariate is dropped.
    threshold = cluster_threshold(comparison, x.shape[0], y.shape[0])

    if is_corr and cov is not None:
        if cov.shape[0] != x.shape[0]:
            raise ValueError("The covariate rows no longer line up with the analysed subjects.")
        bad = np.isnan(cov).any(axis=1)
        if bad.any():
            _note(notes, f"{int(bad.sum())} of {bad.size} subjects have a missing covariate "
                         "value and are dropped from the whole analysis.")
            x, y, cov = x[~bad], y[~bad], cov[~bad]
            subjects = tuple(s for s, b in zip(subjects, bad) if not b)
            if x.shape[0] < 3:
                raise ValueError(f"Only {x.shape[0]} subjects have complete covariates.")

    if not is_corr:
        # A non-finite cell (log of zero power, for example) is an unusable cell, like a missing one.
        x[~np.isfinite(x)] = np.nan
        y[~np.isfinite(y)] = np.nan
    rule = missing.channel_rule(comparison, x, y)
    if not rule.evaluable.any():
        raise ValueError("No channel can be tested: " + rule.reason + ".")
    if is_corr and rule.excluded.size / rule.evaluable.size > 0.10:
        _note(notes, f"{rule.excluded.size} of {rule.evaluable.size} channels were excluded for "
                     "missing data.")
    x[:, ~rule.evaluable] = np.nan
    y[:, ~rule.evaluable] = np.nan
    n_analysed = missing.n_analysed(comparison, x, y)

    stat_comparison, ncov = comparison, 0
    before, fl, cov_info = None, None, None
    if is_corr and cov is not None:
        if comparison == "correlationS":
            # The rank partial correlation is the linear correlation of rank residuals. Ranks are
            # taken once, here; the Freedman-Lane permutation then runs on the rank residuals
            # exactly as on raw residuals for Pearson, and never re-ranks.
            x, y, cov = legacy.tiedrank(x), legacy.tiedrank(y), legacy.tiedrank(cov)
            stat_comparison = "correlationP"
        # Freedman-Lane partial correlation (Method Spec 2026-10-18; Freedman and Lane 1983,
        # Winkler et al. 2014): one residual-forming matrix of [1, covariates] on the final
        # analysed rows, both matrices residualised through it, and the engines re-project
        # every permuted map through it. Permuting x~ against y~ and stopping there (Kennedy)
        # gave family-wise error up to 0.87 at nominal 0.05.
        Rz, k_eff = _residual_forming(cov)
        n_fl = x.shape[0]
        if k_eff < cov.shape[1]:
            _note(notes, f"The {cov.shape[1]} covariate column(s) span only {k_eff} "
                         "dimension(s) beyond the intercept (a duplicated, constant or linearly "
                         "dependent covariate). The analysis controls for that "
                         f"{k_eff}-dimensional space and every degrees-of-freedom statement "
                         f"uses {k_eff} covariate(s).")
        fl_df = n_fl - 2 - k_eff
        if fl_df < 1:
            raise CovariateNoErrorDF(
                f"Partial correlation with {k_eff} covariate(s) on {n_fl} subjects leaves "
                f"{fl_df} error degrees of freedom (n - 2 - covariates). It needs at least 1: "
                "drop covariates or add subjects.")
        conservative = ""
        if fl_df < 5:
            conservative = _conservative_note(fl_df, k_eff, n_fl)
            _note(notes, conservative)
        spearman_caveat = SPEARMAN_CAVEAT if comparison == "correlationS" else ""
        if spearman_caveat:
            notes.append(spearman_caveat)
        before = (x, y)
        x, y = _residualise(x, Rz), _residualise(y, Rz)
        ncov = k_eff
        fl = legacy.FLContext(Rz, k_eff)
        cov_info = CovariateInfo(cov_names, cov.shape[1], k_eff, fl_df,
                                 spearman_caveat=spearman_caveat,
                                 conservative_note=conservative)

    # Channels with nothing to permute are excluded once, here, with a reason. This is decided
    # on the matrices the statistic is computed from, after the missing-data rule.
    if not is_corr:
        flat, why = missing.t_test_no_variation(comparison, x, y)
    elif before is None:
        flat, why = missing.correlation_no_variation(x, y, comparison == "correlationS"), \
            missing.CONSTANT
    else:
        constant = missing.correlation_no_variation(*before, spearman=False)
        flat = constant | missing.explained_by_covariates(before[0], x) \
            | missing.explained_by_covariates(before[1], y)
        why = np.where(constant, missing.CONSTANT, missing.EXPLAINED)
    newly = rule.evaluable & flat
    if newly.any():
        rule = rule.exclude(newly, why)
        for reason in dict.fromkeys(rule.why[newly]):
            idx = np.flatnonzero(newly & (rule.why == reason))
            _note(notes, f"{idx.size} channel(s) are not tested ({reason}): "
                         f"{', '.join(setup.labels[i] for i in idx)}.")
        x[:, newly] = np.nan
        y[:, newly] = np.nan
        n_analysed = missing.n_analysed(comparison, x, y)
        if not rule.evaluable.any():
            raise ValueError("No channel can be tested: " + "; ".join(dict.fromkeys(rule.reasons))
                             + ".")

    # A channel whose observed data separate perfectly (no spread within the groups, or every
    # paired difference identical) has no defined t. It is NaN in the observed map and listed,
    # but stays in every permuted map, where the statistic is 0 at such labellings.
    undefined = np.zeros(rule.evaluable.size, dtype=bool)
    if comparison in missing.T_TESTS:
        undefined = rule.evaluable & (legacy.unpaired_t(x, y, return_undefined=True)[2]
                                      if comparison == "unpairedT"
                                      else legacy.paired_t(x - y, return_undefined=True)[2])
        if undefined.any():
            _note(notes, f"{int(undefined.sum())} channel(s) are not tested "
                         f"({missing.PERFECT_SEPARATION}): "
                         f"{', '.join(setup.labels[i] for i in np.flatnonzero(undefined))}.")
    interpretable = not (is_corr and datatype == "normalize")
    if not interpretable:
        _note(notes, "The whole-head test is not interpretable with datatype 'normalize': each "
                     "subject's channel average is zero after z-scoring, so it correlates "
                     "rounding error. The channel maps are unaffected.")
    return Prepared(comparison, stat_comparison, tail, x, y, setup, rule, n_analysed, threshold,
                    ncov, cov_names if cov_info else (), subjects, behavior_col, notes,
                    interpretable, undefined, fl, cov_info)


def run_legacy(params: dict, rng: np.random.Generator | None = None,
               perms_tfce=None, perms_cluster=None) -> LegacyResults:
    """Run a t-test or correlation analysis end to end (no files are written here).

    The TFCE and cluster tests draw their permutations separately, as in MATLAB;
    ``perms_tfce`` and ``perms_cluster`` replace those draws.
    """
    prep = prepare(params)
    rng = rng if rng is not None else np.random.default_rng(params.get("seed"))
    # The engines' own default is every relabelling (2^n sign flips), which is unusable at
    # ordinary sample sizes; the pipeline asks for 5000 unless told otherwise.
    n_perm = params.get("permutations", DEFAULT_PERMUTATIONS)
    g = legacy.global_test(prep.x, prep.y, prep.stat_comparison, prep.tail, prep.ncov)
    T = legacy.single_threshold_tfce(prep.x, prep.y, prep.setup.graph, prep.stat_comparison,
                                     prep.tail, ALPHA, n_perm, TFCE_E, TFCE_H, rng, perms_tfce,
                                     fl=prep.fl)
    C = legacy.cluster_analysis(prep.x, prep.y, prep.threshold, prep.setup.graph,
                                prep.stat_comparison, prep.tail, n_perm, rng, perms_cluster,
                                fl=prep.fl)
    return LegacyResults(prep, T, C, g, ALPHA)


# --------------------------------------------------------------------------- GLM presets

GLM_PRESETS = ("anova1", "ancova", "regression", "rmanova", "mixed2way")
_ROLE_KEYS = ("group_col", "condition_col", "subject_col", "predictor_col", "covariate_cols")


@dataclass
class GLMResults:
    preset: str
    loops: list                  # glm.Loop, in run order, each with its result
    setup: ChannelSetup
    meta: object                 # the metadata columns as a DataFrame
    alpha: float
    notes: list[str] = field(default_factory=list)

    def loop(self, role: str):
        return next(lp for lp in self.loops if lp.role == role)

    @property
    def effects(self) -> list:
        """The loops that are effects in their own right (not post-hoc contrasts)."""
        return [lp for lp in self.loops if "/posthoc:" not in lp.role]


def _one_row_per_subject(meta, cols, preset):
    """Between-subject presets permute whole rows, which is valid only with one row per subject."""
    names = [str(c) for c in meta.columns]
    col = cols.get("subject_col") if cols.get("subject_col") in names else tables.subject_column(names)
    if col is None:
        return
    counts = meta[col].value_counts(sort=False)
    if (counts > 1).any():
        first = counts[counts > 1].index[0]
        raise ValueError(
            f"Preset '{preset}' is a between-subject test and needs one row per subject, but "
            f"subject '{first}' appears in {counts[first]} rows. Use 'rmanova' for repeated "
            "conditions.")


def prepare_glm(params: dict):
    """Read the single GLM table: (Y, meta, column roles, channel setup)."""
    preset = params["comparison"].lower()
    if preset not in GLM_PRESETS:
        raise ValueError(f"unsupported preset {preset!r}")
    circ_pipeline.tier1_guard(params)
    meta_cols = [str(c) for c in np.atleast_1d(params.get("meta_cols", []))]
    if not meta_cols:
        raise ValueError("params['meta_cols'] must list the non-channel columns.")
    path = params.get("data_file") or params.get("data1_file")
    table = tables.read_table(path, params.get("data_sheet") or params.get("data1_sheet"))
    missing_cols = [c for c in meta_cols if c not in table.columns]
    if missing_cols:
        raise ValueError(f"meta column(s) not in the file: {', '.join(missing_cols)}")
    channel_cols = [str(c) for c in table.columns if str(c) not in meta_cols]
    if not channel_cols:
        raise ValueError("No channel columns are left after removing the meta columns.")
    meta = table[meta_cols].reset_index(drop=True)
    # As in MATLAB, the transform runs over every data column before channels are selected.
    power = _transform(tables.numeric_matrix(table[channel_cols], "The data file"),
                       params.get("datatype", "absolute"))
    setup = setup_channels(params.get("channels", "egi"), [power], channel_cols)
    cols = {}
    for k in _ROLE_KEYS:
        v = params.get(k)
        if v is not None and np.size(v):
            cols[k] = [str(c) for c in np.atleast_1d(v)] if k == "covariate_cols" else str(v)
    if preset == "mixed2way":
        absent = [k for k in ("group_col", "condition_col", "subject_col") if k not in cols]
        if absent:
            raise ValueError(f"mixed2way needs {', '.join(absent)}.")
    elif preset in ("anova1", "ancova", "regression"):
        _one_row_per_subject(meta, cols, preset)
    return np.array(setup.data[0], dtype=float), meta, cols, setup


def run_glm(params: dict, rng: np.random.Generator | None = None, perms=None) -> GLMResults:
    """Run a GLM preset end to end (no files are written here). ``perms`` is one row-index
    matrix per permutation loop, in run order."""
    from . import glm

    Y, meta, cols, setup = prepare_glm(params)
    rng = rng if rng is not None else np.random.default_rng(params.get("seed"))
    loops = glm.run_preset(params["comparison"], Y, meta, setup.graph,
                           n_perm=params.get("permutations", DEFAULT_PERMUTATIONS), rng=rng, perms=perms,
                           alpha=ALPHA, E=TFCE_E, H=TFCE_H, **cols)
    notes = [(f"S{pair[1:]} was not run: {why}." if pair.startswith(glm.SIMPLE_EFFECT)
              else f"Post-hoc {lp.role}: {pair} was not run: {why}.")
             for lp in loops for pair, why in lp.skipped]
    for lp in loops:
        groups: dict[str, list[str]] = {}
        for i, why in zip(lp.result.excluded_channels, lp.result.excluded_reasons):
            if why != glm.MISSING:
                groups.setdefault(why, []).append(setup.labels[i])
        notes += [f"{lp.role}: {len(chans)} channel(s) are not tested ({why}): "
                  f"{', '.join(chans)}." for why, chans in groups.items()]
    notes += [f"Post-hoc {lp.role.replace('/posthoc:', ': ')} left out subject(s) "
              f"{', '.join(lp.dropped)}, who have only one of the two levels." for lp in loops
              if lp.dropped]
    return GLMResults(params["comparison"].lower(), loops, setup, meta, ALPHA, notes)


def run(params: dict, rng: np.random.Generator | None = None):
    """Run whichever analysis ``params['comparison']`` names."""
    n_perm = params.get("permutations")
    if n_perm is not None and (int(n_perm) != n_perm or n_perm < 20):
        raise ValueError(f"permutations must be a whole number of at least 20, got {n_perm!r}; "
                         "with fewer, no p-value can reach 0.05.")
    comparison = params["comparison"]
    if comparison in circ_pipeline.RETIRED:
        raise ValueError(circ_pipeline.retired_message(comparison))
    if comparison in circ_pipeline.COMPARISONS:
        return circ_pipeline.run_circ(params, rng)
    if comparison in legacy.COMPARISONS:
        return run_legacy(params, rng)
    return run_glm(params, rng)
