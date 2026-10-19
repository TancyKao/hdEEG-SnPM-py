"""Circular (phase) tier from input files to corrected maps; port of ``core_snpm_circ.m``.

Analyses (``params['comparison']``, MATLAB's keys):

- ``circ_phase_group``: covariate-adjusted Hotelling T^2 on the (cos, sin) embedding, two groups.
- ``circ_phase_group_u2``: Watson's U^2, same design; cannot adjust for event count.
- ``circ_corrAngLinear``: circular-linear F of a linear measure on [1, cos, sin], paired by subject.

What makes it correct, in MATLAB's words (header of ``core_snpm_circ.m``):

1. The angle matrix is never permuted. Hotelling permutes the stacked response ``[cos, sin]``
   (Freedman-Lane, one row index for every column); the circular-linear test permutes the
   measure; U^2 permutes the group labels against the fixed angles.
2. The event-count covariate is per channel (``circ_count_covariate_mode``, default
   ``per_channel``), fitted once before the loop.
3. The >= 8-per-group estimability gate is computed ONCE from the observed labels and held fixed;
   the first permutation asserts that the observed and permuted maps share one channel set
   (``CircGateDrift``).
4. No whole-head test. A descriptive panel (mean direction, circular SD, Rayleigh) is reported
   instead and is labelled no-inference.

Locked parameters: ``tail`` must be ``both`` and ``datatype`` ``absolute``; ``circ_units`` and
``circ_convention`` are required with no default.

Also here: ``tier1_guard``, which refuses an unadjusted analysis of the Tier-1 UNSIGNED linearised
measure (``snpm_circ_tier1_guard.m``); ``pipeline`` runs it before every non-circular analysis.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import circ, glm
from . import perm as _perm
from .io import tables
from .montage import ChannelSetup, setup_channels

COMPARISONS = ("circ_phase_group", "circ_phase_group_u2", "circ_corrAngLinear")
RETIRED = ("circ_wheeler_watson_Test", "circ_WatsonsU2Test")
GATE_MIN = 8
TFCE_DH, TFCE_DH_U2 = 0.1, 0.005
CLUSTER_STATISTIC = "mass (sum stat / n channels)"

STAT_LABELS = {"circ_phase_group": "Hotelling T^2 on (cos,sin), two groups",
               "circ_phase_group_u2": "Watson U^2, two groups",
               "circ_corrAngLinear": "Circular-linear association (F)"}
STATISTICS = {"circ_phase_group": "hotelling_F", "circ_phase_group_u2": "watson_U2",
              "circ_corrAngLinear": "circlinear_F"}

# results_struct.excluded_channels.reason, one string for every excluded channel
EXCLUSION_REASON = ("not estimable: fewer than 8 subjects per group, incomplete column under a "
                    "per-channel covariate, or log(count) collinear with group")
# per-channel reasons, one per exclusion route (the export script's words)
REASON_BELOW = "fewer than 8 subjects per group"   # also circ_corrAngLinear, as MATLAB words it
REASON_COLLINEAR = "log(count) collinear with group"
REASON_INCOMPLETE = "incomplete column under the per-channel covariate"

CLUSTER_NOTE = ("Cluster statistic: MASS (sum of the statistic / n channels). This differs from "
                "the FieldTrip maxsize used by Helfrich et al.")
P_NOTE = ("Note: the parametric per-channel p is a cluster-forming and descriptive quantity "
          "only. For Watson U^2 at n=8 per group the exact enumerated null gives 0.1553 / 0.1787 "
          "/ 0.2412 at alpha .10 / .05 / .01 against the asymptotic 0.152 / 0.187 / 0.268 -- "
          "good agreement at .10 and .05, conservative in the far tail. The inferential p is "
          "always the permutation p.")


class CircGateDrift(RuntimeError):
    """The observed and permuted maps are not defined on one channel set
    (MATLAB ``core_snpm:circGateDrift``)."""


def retired_message(key: str) -> str:
    return (f"params.comparison = '{key}' is RETIRED: its engine has been deleted and it is not "
            "a supported analysis. Replace it with one of the Tier-2 circular analyses:\n"
            "  'circ_phase_group'      two independent groups, event-count-adjusted Hotelling "
            "T^2 on the (cos,sin) embedding. PRIMARY replacement for a two-group phase "
            "comparison (this is what circ_wheeler_watson_Test was used for).\n"
            "  'circ_phase_group_u2'   same design, Watson's U^2 omnibus. Replacement for "
            "circ_WatsonsU2Test; note it CANNOT adjust for event count.\n"
            "  'circ_corrAngLinear'    angle vs a linear measure, paired by subject.\n"
            "All three route to core_snpm_circ and REQUIRE params.circ_units and "
            "params.circ_convention; the two phase-group analyses also require "
            "params.count1_file / params.count2_file. See core_snpm_circ.m for the full "
            "parameter list.")


# --------------------------------------------------------------------------- reading

@dataclass
class _Wide:
    subj: np.ndarray | None      # normalised subject IDs, or None without a subject column
    cols: list[str]
    M: np.ndarray


def _read_wide(path, sheet=None) -> _Wide:
    """``read_wide``: the first column whose name contains 'subject' is the ID; the rest is data."""
    t = tables.read_table(path, sheet)
    col = tables.subject_column(t.columns)
    subj = None if col is None else tables.subject_ids(t[col])
    data = t.drop(columns=col) if col is not None else t
    bad = [str(c) for c in data.columns if not (pd.api.types.is_numeric_dtype(data[c])
                                                 or pd.api.types.is_bool_dtype(data[c]))]
    if bad:
        raise circ.CircInputError("circNonNumeric", f"Non-numeric data columns in {path}.")
    return _Wide(subj, [str(c) for c in data.columns], data.to_numpy(dtype=float))


def _align_columns(S: _Wide, ref_cols, what) -> _Wide:
    """Exact SET match on channel names (order free); reorder into the reference order."""
    missing = [c for c in ref_cols if c not in S.cols]
    extra = sorted(set(S.cols) - set(ref_cols))
    if missing or extra:
        raise circ.CircInputError(
            "circColumnMismatch",
            f"{what} must have exactly the same channel columns as the reference angle file "
            f"(order does not matter). Missing: {{{', '.join(missing)}}}. Unexpected: "
            f"{{{', '.join(extra)}}}.")
    pos = {c: i for i, c in enumerate(S.cols)}
    return _Wide(S.subj, list(ref_cols), S.M[:, [pos[c] for c in ref_cols]])


def _align_subjects(S: _Wide, ref_ids, what) -> _Wide:
    """Exact SET match on subject IDs after sub%03d normalisation; reorder."""
    if ref_ids is None:
        raise circ.CircInputError(
            "circSubjectColumn",
            f"The angle file has no Subject column, so {what} cannot be matched to it. Both "
            "files need a subject identifier column.")
    if S.subj is None:
        raise circ.CircInputError("circSubjectColumn", f"{what} has no Subject column.")
    ids = list(S.subj)
    missing = [s for s in ref_ids if s not in ids]
    extra = sorted(set(ids) - set(ref_ids))
    if missing or extra:
        raise circ.CircInputError(
            "circSubjectMismatch",
            f"{what} must contain exactly the same subjects as its angle file. Missing: "
            f"{{{', '.join(missing)}}}. Unexpected: {{{', '.join(extra)}}}.")
    pos = {s: i for i, s in enumerate(ids)}
    return _Wide(np.asarray(ref_ids, dtype=object), S.cols, S.M[[pos[s] for s in ref_ids]])


def _match_measure(A1, D1: _Wide, D2: _Wide, measure_col):
    """circ_corrAngLinear: angles plus one linear measure column, paired by subject (subjects
    intersected in angle-file order)."""
    if D1.subj is None or D2.subj is None:
        raise circ.CircInputError("circSubjectColumn", "circ_corrAngLinear needs a Subject column "
                                  "in both the angle and the measure file.")
    col = measure_col or ""
    if not col:
        if len(D2.cols) != 1:
            raise circ.CircInputError(
                "circMeasureColumn",
                f"The measure file has {len(D2.cols)} non-subject columns {{{', '.join(D2.cols)}}}; "
                "name the one to use in params.measure_col.")
        col = D2.cols[0]
    if col not in D2.cols:
        raise circ.CircInputError("circMeasureColumn",
                                  f"params.measure_col '{col}' is not a column of the measure file.")
    k = D2.cols.index(col)
    pos2: dict[str, int] = {}
    for j, s in enumerate(D2.subj):
        pos2.setdefault(s, j)
    seen, i1, i2 = set(), [], []
    for i, s in enumerate(D1.subj):
        if s in pos2 and s not in seen:
            seen.add(s)
            i1.append(i)
            i2.append(pos2[s])
    if len(i1) < GATE_MIN:
        raise circ.CircInputError("circTooFewSubjects",
                                  f"Only {len(i1)} subjects match between the angle and measure "
                                  f"files (need >= {GATE_MIN}).")
    return A1[i1], D2.M[i2, k], D1.subj[i1], col, np.asarray(i2)


def _drop_missing_measure(A, y, subjects):
    """A subject with no finite measure value (NaN or Inf) has no response row at ANY channel,
    so it is dropped whole, before channel setup, gate, inversion check and descriptives. Left
    in, its NaN would sit in the Freedman-Lane residual and move with every permutation, so each
    permuted map would delete a different subject's row (biostatistician ruling 2026-10-19;
    MATLAB ``core_snpm:circMeasureNaNRows``). Missing angles stay with their row and are deleted
    per channel on this fixed row set."""
    keep = np.isfinite(y)
    dropped = [str(s) for s in subjects[~keep]]
    if dropped:
        warnings.warn(f"{len(dropped)} of {keep.size} matched subjects have a missing (or "
                      "non-finite) value for the linear measure and cannot enter the "
                      "circular-linear test at any channel; they are dropped from the analysis: "
                      f"{', '.join(dropped)}.", stacklevel=3)
        A, y, subjects = A[keep], y[keep], subjects[keep]
    if y.size < GATE_MIN:
        raise circ.CircInputError("circTooFewSubjects",
                                  f"Only {y.size} subjects have both angles and a finite measure "
                                  f"value (need >= {GATE_MIN}).")
    return A, y, subjects, dropped


def _get(params, key, default=None):
    v = params.get(key)
    return default if v is None or (isinstance(v, str) and v == "") else v


# --------------------------------------------------------------------------- prepare

@dataclass
class CircPrepared:
    """Everything decided before the first permutation."""

    comparison: str
    setup: ChannelSetup
    imp: dict                    # import record of angle file 1
    A_imported: np.ndarray       # after import and channel selection, before the count rule
    A: np.ndarray                # as analysed
    gi: np.ndarray | None        # 1 = group A (file 1), 0 = group B; None for circular-linear
    y: np.ndarray | None         # circular-linear measure
    subjects: np.ndarray
    counts: np.ndarray | None
    logC: np.ndarray | None
    usecov: np.ndarray
    collinear: np.ndarray
    covariate: dict
    gate0: np.ndarray
    gate: np.ndarray
    complete: np.ndarray
    evaluable: np.ndarray
    n_group1: np.ndarray | None
    n_group2: np.ndarray | None
    n_finite: np.ndarray
    inversion: dict
    excluded: np.ndarray         # 0-based, analysed order
    excluded_reasons: list[str]  # one per excluded channel
    n1g: int
    n2g: int
    measure_col: str = ""
    dropped_subjects: list[str] = field(default_factory=list)   # missing measure value
    notes: list[str] = field(default_factory=list)

    @property
    def measure_nan_removed(self) -> int:
        return len(self.dropped_subjects)

    @property
    def is_group(self) -> bool:
        return self.comparison != "circ_corrAngLinear"

    @property
    def statistic(self) -> str:
        return STATISTICS[self.comparison]

    @property
    def tfce_dh(self) -> float:
        return TFCE_DH_U2 if self.comparison == "circ_phase_group_u2" else TFCE_DH

    @property
    def per_channel_n(self) -> np.ndarray:
        return np.isfinite(self.A).sum(axis=0)


def _check_locked(params):
    tail = _get(params, "tail")
    if tail is not None and str(tail).lower() != "both":
        raise circ.CircInputError(
            "circTailNotSupported",
            f"params.tail = '{tail}' is not defined for a circular analysis. The Hotelling F, "
            "Watson U^2 and circular-linear F are all non-negative omnibus statistics with a "
            "single upper tail; use tail = 'both'.")
    dt = _get(params, "datatype")
    if dt is not None and str(dt).lower() != "absolute":
        raise circ.CircInputError(
            "circDatatypeNotSupported",
            f"params.datatype = '{dt}' is undefined on a circular domain (log10 of a radian goes "
            "complex; a cross-channel z-score of an angle is meaningless). Use 'absolute'.")


def _load_counts(params, D1: _Wide, D2: _Wide, kept) -> np.ndarray:
    if not _get(params, "count1_file") or not _get(params, "count2_file"):
        raise circ.CircInputError(
            "circCountsRequired",
            f"params.count1_file and params.count2_file are REQUIRED for "
            f"'{params['comparison']}'. A subject's preferred phase is estimated from their "
            "detected events, so its precision -- and therefore the resultant length the "
            "Hotelling test reads -- depends on how many events there were. Without the count "
            "there is no way to tell a genuine phase difference from a detection-rate "
            "difference. Supply one wide file per group: subjects x channels, same subject "
            "column and same channel columns as the corresponding angle file.")
    C1 = _read_wide(params["count1_file"], _get(params, "count1_sheet"))
    C2 = _read_wide(params["count2_file"], _get(params, "count2_sheet"))
    C1 = _align_subjects(_align_columns(C1, D1.cols, "Group A count file"), D1.subj,
                         "Group A count file")
    C2 = _align_subjects(_align_columns(C2, D1.cols, "Group B count file"), D2.subj,
                         "Group B count file")
    return np.vstack([C1.M, C2.M])[:, kept]


def prepare_circ(params: dict) -> CircPrepared:
    """Read the files named in ``params`` and fix everything the permutation loop holds constant:
    import, channels, count rule and covariate, convention check, the gate and the channel set."""
    comparison = params["comparison"]
    if comparison not in COMPARISONS:
        raise circ.CircInputError("circUnknownComparison",
                                  f"core_snpm_circ does not handle comparison '{comparison}'.")
    _check_locked(params)
    is_u2 = comparison == "circ_phase_group_u2"
    is_grp = comparison != "circ_corrAngLinear"
    notes: list[str] = []

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        D1 = _read_wide(params["data1_file"], _get(params, "data1_sheet"))
        D2 = _read_wide(params["data2_file"], _get(params, "data2_sheet"))
        opts = {"units": _get(params, "circ_units", ""),
                "convention": _get(params, "circ_convention", ""),
                "zero_offset_deg": _get(params, "circ_zero_offset_deg"),
                "resolution_gate": "error" if is_u2 else "warn"}
        A, imp = circ.import_angles(D1.M, label="Angles file 1", **opts)
        y, measure_col, measure_dropped = None, "", []
        if is_grp:
            D2 = _align_columns(D2, D1.cols, "Group B angle file")
            A2, _ = circ.import_angles(D2.M, label="Angles file 2", **opts)
            n1g, n2g = A.shape[0], A2.shape[0]
            A = np.vstack([A, A2])
            gi = np.r_[np.ones(n1g), np.zeros(n2g)]
            subjects = np.concatenate([D1.subj if D1.subj is not None else
                                       np.array([f"A{i + 1}" for i in range(n1g)], dtype=object),
                                       D2.subj if D2.subj is not None else
                                       np.array([f"B{i + 1}" for i in range(n2g)], dtype=object)])
        else:
            A, y, subjects, measure_col, _ = _match_measure(A, D1, D2,
                                                            _get(params, "measure_col", ""))
            A, y, subjects, measure_dropped = _drop_missing_measure(A, y, subjects)
            gi, n1g, n2g = None, A.shape[0], 0

        setup = setup_channels(_get(params, "channels", "egi"), [A], D1.cols)
        A = np.array(setup.data[0], dtype=float)
        A_imported = A.copy()
        n_ch = A.shape[1]
        if setup.neighbors.shape[0] != n_ch:
            raise circ.CircInputError("circChannelMismatch",
                                      f"Selected {n_ch} channels but the neighbour matrix has "
                                      f"{setup.neighbors.shape[0]} rows.")

        counts = logC = None
        usecov = np.zeros(n_ch, bool)
        collinear = np.zeros(n_ch, bool)
        cov_report = {"mode": "none", "n_dropped_constant": 0, "n_collinear": 0,
                      "pb_max": np.nan, "pb_median": np.nan, "n_cells_no_count": 0,
                      "frac_cells_no_count": 0.0}
        if is_grp:
            counts = _load_counts(params, D1, D2, setup.data_columns)
            cc = circ.count_covariate(
                A, gi, counts, mode=_get(params, "circ_count_covariate_mode", "per_channel"),
                collinearity_max=float(_get(params, "circ_collinearity_max", 0.8)),
                labels=setup.labels)
            A, logC, usecov, collinear, cov_report = cc.A, cc.logC, cc.usecov, cc.collinear, \
                cc.report

        # convention check: pooled over both groups (label-invariant), frontal gated channels
        fin = np.isfinite(A)
        if is_grp:
            n_g1 = (fin & (gi == 1)[:, None]).sum(axis=0)
            n_g2 = (fin & (gi == 0)[:, None]).sum(axis=0)
            gate0 = (n_g1 >= GATE_MIN) & (n_g2 >= GATE_MIN) & ~collinear
        else:
            n_g1 = n_g2 = None
            gate0 = (fin.sum(axis=0) >= GATE_MIN) & ~collinear
        inversion = circ.check_inversion(A, gate0, setup.xyz[:, 0], imp)

        # estimability gate, computed ONCE from the observed labels and held fixed
        gate = gate0.copy()
        if is_grp:
            ex = circ.hotelling(A, gi)                       # the documented idiom
            gate &= (ex.n1 >= GATE_MIN) & (ex.n2 >= GATE_MIN)
            n_fin = n_g1 + n_g2
            below = ~((n_g1 >= GATE_MIN) & (n_g2 >= GATE_MIN))
        else:
            n_fin = (fin & np.isfinite(y)[:, None]).sum(axis=0)
            gate &= n_fin >= GATE_MIN
            below = n_fin < GATE_MIN
        # a channel using the per-channel covariate must be complete (Freedman-Lane fits the
        # nuisance over whole columns)
        complete = fin.all(axis=0)
        evaluable = gate & (~usecov | complete)
        if not evaluable.any():
            raise circ.CircInputError(
                "circNoEvaluableChannels",
                "No channel is estimable: every channel fails the >=8-per-group gate, the "
                "completeness requirement, or the log(count) collinearity check. "
                f"{int(collinear.sum())}/{n_ch} channels were collinear with group.")
        incomplete = usecov & ~complete
        if not np.array_equal(~evaluable, below | collinear | incomplete):
            raise CircGateDrift("The excluded channels are not explained by the three exclusion "
                                "routes; the gate is inconsistent.")
        excluded = np.flatnonzero(~evaluable)
        reasons = ["; ".join(r for r, hit in ((REASON_BELOW, below[k]),
                                              (REASON_COLLINEAR, collinear[k]),
                                              (REASON_INCOMPLETE, incomplete[k])) if hit)
                   for k in excluded]

        if is_u2 and usecov.any():
            warnings.warn(
                "Watson U^2 cannot take the event-count covariate, so this map is UNADJUSTED for "
                "event count (max |point-biserial(log count, group)| = "
                f"{cov_report['pb_max']:.2f}). Use comparison = 'circ_phase_group' if that "
                "association is material.")

    for w in caught:
        if "value for the linear measure" not in str(w.message):
            notes.append(str(w.message))
        warnings.warn(w.message, w.category, stacklevel=2)
    if measure_dropped:      # MATLAB's results-text line; shown as a notice in the report
        notes.append(f"Subjects dropped for a missing measure value: {len(measure_dropped)} "
                     f"({', '.join(measure_dropped)}); they are excluded at every channel")
    if is_grp:
        notes.append(
            f"Count/group confound: point-biserial r median {cov_report['pb_median']:.2f}, "
            f"max |r| {cov_report['pb_max']:.2f}; covariate dropped at "
            f"{cov_report['n_dropped_constant']} channels (constant), "
            f"{cov_report['n_collinear']} channels non-estimable (collinear)")
    groups: dict[str, list[str]] = {}
    for k, r in zip(excluded, reasons):
        groups.setdefault(r, []).append(setup.labels[k])
    notes += [f"{len(ch)} channel(s) are not tested ({r}): {', '.join(ch)}."
              for r, ch in groups.items()]

    return CircPrepared(
        comparison=comparison, setup=setup, imp=imp, A_imported=A_imported, A=A, gi=gi, y=y,
        subjects=np.asarray(subjects, dtype=object), counts=counts, logC=logC, usecov=usecov,
        collinear=collinear, covariate=cov_report, gate0=gate0, gate=gate, complete=complete,
        evaluable=evaluable, n_group1=n_g1, n_group2=n_g2, n_finite=n_fin, inversion=inversion,
        excluded=excluded, excluded_reasons=reasons, n1g=int(n1g), n2g=int(n2g),
        measure_col=measure_col, dropped_subjects=measure_dropped, notes=notes)


# --------------------------------------------------------------------------- permutation

@dataclass
class CircEngine:
    """The observed maps and how one permuted map is produced from a row index ``ix``."""

    stat: np.ndarray
    p: np.ndarray
    perm_map: object             # ix -> (stat, p)
    perm_type: str               # 'free' (Freedman-Lane) or 'labels' (U^2)
    fl_fit: np.ndarray | None = None
    fl_resid: np.ndarray | None = None
    rho: np.ndarray | None = None


def _anglinear_fl(y):
    """Freedman-Lane context of the circular-linear test, the intercept the only nuisance. Subjects
    with a missing measure were dropped up front, so the permuted response never carries a NaN."""
    if not np.isfinite(y).all():
        raise circ.CircInputError("circMeasureNaN", "The measure reached the permutation with a "
                                  "non-finite value.")
    mu = y.mean()
    return np.full((y.size, 1), mu), (y - mu)[:, None]


def build_engine(prep: CircPrepared) -> CircEngine:
    A, ev = prep.A, prep.evaluable
    if prep.comparison == "circ_phase_group":
        hp = circ.HotellingPerm(prep.gi, prep.logC, prep.usecov, ev)
        stat, p, _ = hp.evaluate(np.cos(A), np.sin(A))
        ctx = circ.fl_context(A, prep.logC, prep.usecov, ev)

        def perm_map(ix):
            yc, ys = circ.permuted_embedding(ctx, ix)
            s, pp, _ = hp.evaluate(yc, ys)
            return s, pp

        return CircEngine(stat, p, perm_map, "free", ctx.fit, ctx.resid)
    if prep.comparison == "circ_phase_group_u2":
        r = circ.watson_u2(A, prep.gi, valid_channels=ev)

        def perm_map(ix):
            rp = circ.watson_u2(A, prep.gi[ix], valid_channels=ev)
            return rp.U2, rp.p

        return CircEngine(r.U2, r.p, perm_map, "labels")
    r = circ.corr_anglinear(prep.y, A, valid_channels=ev)
    fit, resid = _anglinear_fl(prep.y)

    def perm_map(ix):
        rp = circ.corr_anglinear((fit + resid[ix])[:, 0], A, valid_channels=ev)
        return rp.F, rp.p

    return CircEngine(r.F, r.p, perm_map, "free", fit, resid, r.rho)


def _assert_gate_fixed(evaluable, obs, perm_stat):
    a, b = np.isnan(obs[evaluable]), np.isnan(np.asarray(perm_stat)[evaluable])
    if not np.array_equal(a, b):
        raise CircGateDrift(
            "The observed and permuted statistic maps are defined on different channel sets "
            f"({int((~a).sum())} observed vs {int((~b).sum())} permuted finite channels inside "
            "the gate). The max-statistic null would not be calibrated.")


@dataclass
class CircResults:
    prepared: CircPrepared
    engine: CircEngine
    result: glm.CorrectionResult
    descriptive: dict
    alpha: float
    seed: int | None = None
    tier: str = "circ"

    @property
    def setup(self) -> ChannelSetup:
        return self.prepared.setup

    @property
    def notes(self) -> list[str]:
        return self.prepared.notes

    @property
    def uncorrsigch(self):
        with np.errstate(invalid="ignore"):
            return np.flatnonzero(self.result.p_real <= self.alpha)

    @property
    def correctTFCEsigch(self):
        with np.errstate(invalid="ignore"):
            return np.flatnonzero(self.result.p_correctedTFCE <= self.alpha)

    @property
    def SnPMsigch(self):
        sig = [c.channels for c in self.result.clusters if c.p <= self.alpha]
        return np.concatenate(sig).astype(int) if sig else np.empty(0, dtype=int)


def run_circ(params: dict, rng: np.random.Generator | None = None, perms=None) -> CircResults:
    """Run a circular analysis end to end (no files are written here).

    ``perms`` is an (n_perm, n_subjects) 0-based row-index matrix for the one permutation loop:
    Freedman-Lane rows for Hotelling and circular-linear, label orders for U^2. Without it the
    rows are drawn from ``rng``.
    """
    from .pipeline import ALPHA, DEFAULT_PERMUTATIONS, TFCE_E, TFCE_H

    prep = prepare_circ(params)
    eng = build_engine(prep)
    n = prep.A.shape[0]
    if perms is None:
        rng = rng if rng is not None else np.random.default_rng(params.get("seed"))
        n_perm = int(params.get("permutations", DEFAULT_PERMUTATIONS))
        perms = _perm.freedman_lane_indices(n, n_perm, "free", np.ones(n), rng)
    perms = np.asarray(perms, dtype=int)
    if perms.ndim != 2 or perms.shape[1] != n:
        raise ValueError(f"perms must be (n_perm, {n}); got {perms.shape}.")
    first = [True]

    def perm_stat(i):
        s, pp = eng.perm_map(perms[i])
        if first[0]:
            _assert_gate_fixed(prep.evaluable, eng.stat, s)
            first[0] = False
        return s, pp

    reasons = np.full(prep.A.shape[1], "", dtype=object)
    reasons[prep.excluded] = prep.excluded_reasons
    res = glm.perm_correction(eng.stat, eng.p, perm_stat, prep.setup.graph, perms.shape[0], "F",
                              prep.evaluable, ALPHA, TFCE_E, TFCE_H, prep.tfce_dh,
                              reasons=reasons)
    res.perms = perms
    desc = circ.descriptive(prep.A, prep.gi if prep.is_group else None, prep.evaluable)
    return CircResults(prep, eng, res, desc, ALPHA, params.get("seed"))


def effect_label(prep: CircPrepared) -> str:
    """``results_struct.glm.effect_label``."""
    return (f"{STAT_LABELS[prep.comparison]} | zero convention {prep.imp['convention']} "
            f"(rotation {prep.imp['rotation_deg']:+.1f} deg) | count covariate: "
            f"{prep.covariate['mode']}")


# --------------------------------------------------------------------------- Tier-1 guard

def _read_stamp(path, sheet):
    """The reserved ``circ_provenance`` column's first value decoded, or None."""
    if not isinstance(path, (str, Path)) or not str(path) or not Path(path).exists():
        return None
    p = Path(path)
    try:
        if sheet in (None, "", "CSV File") and p.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
            if "circ_provenance" not in pd.read_csv(p, nrows=0).columns:
                return None
            col = pd.read_csv(p, usecols=["circ_provenance"], dtype=str)["circ_provenance"]
        else:
            t = pd.read_excel(p, sheet_name=0 if sheet in (None, "", "CSV File") else sheet,
                              dtype=str)
            if "circ_provenance" not in t.columns:
                return None
            col = t["circ_provenance"]
    except Exception as err:      # noqa: BLE001 - any unreadable file only skips the guard
        warnings.warn(f"Could not inspect '{path}' for a circular provenance stamp ({err}); the "
                      "Tier-1 unsigned-measure guard was NOT applied to it.", stacklevel=3)
        return None
    if col.empty or pd.isna(col.iloc[0]) or not str(col.iloc[0]):
        return None
    stamp = {}
    for kv in str(col.iloc[0]).split("|"):
        t = kv.split("=")
        if len(t) >= 2:
            stamp[t[0].strip()] = "=".join(t[1:]).strip()
    return stamp


def tier1_guard(params: dict) -> dict:
    """Refuse an unadjusted analysis of the Tier-1 UNSIGNED linearised measure; port of
    ``snpm_circ_tier1_guard.m``.

    A file written by ``circ.linearise_table`` carries a provenance stamp. When it says the
    data are ``circ_unsigned_distance`` the analysis must be ``ancova`` or ``regression`` with
    the event count among ``covariate_cols`` (``circ_count_col`` names it explicitly).
    Unadjusted, that measure ran a Type-I rate of 0.280 against a nominal 0.05 under an event-count
    imbalance. The signed measure is not guarded.
    """
    info = {"checked": False, "stamp": None, "files": []}
    stamp = None
    for key in ("data_file", "data1_file", "data2_file"):
        f = params.get(key)
        if not f:
            continue
        s = _read_stamp(f, params.get(key.replace("_file", "_sheet")))
        if s:
            stamp = s
            info["files"].append(str(f))
    if not stamp:
        return info
    info.update(checked=True, stamp=stamp)
    if stamp.get("requires_count_covariate", "").strip() != "1":
        return info
    measure = stamp.get("measure", "circ_unsigned_distance")
    cfile = stamp.get("count_file") or ""
    cfile = cfile if cfile and cfile != "none" else "the event-count column"
    preset = str(params.get("comparison", ""))
    if preset not in ("ancova", "regression"):
        raise circ.CircInputError(
            "circUnsignedNeedsCountCovariate",
            f"This file is the Tier-1 UNSIGNED circular measure ({measure}, from "
            f"snpm_circ_linearise) but '{preset}' has no covariate slot, so the event count "
            "cannot be adjusted for. |deviation from the up-state peak| is folded at zero: a "
            "subject with fewer events has a noisier angle and therefore a systematically LARGER "
            "value, so any group difference in event count manufactures a group difference in "
            "the measure (Type-I rate 0.280 against a nominal 0.05 under an event- count "
            "imbalance). Use one of the TWO legal presets, 'ancova' or 'regression', with the "
            f"event count ({cfile}) in params.covariate_cols. Alternatively re-run "
            "snpm_circ_linearise with measure = 'signed', which is symmetric about the pooled "
            "mean direction and needs no count adjustment.")
    covs = params.get("covariate_cols")
    covs = [] if covs is None else [str(c) for c in np.atleast_1d(covs) if str(c)]
    if not _has_count_covariate(covs, params, stamp):
        raise circ.CircInputError(
            "circUnsignedNeedsCountCovariate",
            f"This file is the Tier-1 UNSIGNED circular measure ({measure}, from "
            f"snpm_circ_linearise). Preset '{preset}' does have a covariate slot, but "
            f"params.covariate_cols {{{', '.join(covs)}}} does not contain the event-count column "
            f"(expected something matching '{cfile}', or set params.circ_count_col explicitly). "
            "Without it the measure is not calibrated: the folded deviation rises with "
            "estimation noise, so a group difference in event count alone produces a Type-I "
            "rate of 0.280 against a nominal 0.05. The two legal presets are 'ancova' and "
            "'regression', both WITH the count covariate.")
    return info


def _has_count_covariate(covs, params, stamp) -> bool:
    if not covs:
        return False
    low = [c.lower() for c in covs]
    if params.get("circ_count_col"):
        return str(params["circ_count_col"]).lower() in low
    if any("count" in c for c in low):
        return True
    base = Path(stamp.get("count_file", "")).stem if stamp.get("count_file") else ""
    return bool(base) and base.lower() in low
