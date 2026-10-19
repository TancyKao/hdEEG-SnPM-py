"""Mixed linear model tier: from the long-format file to corrected maps.

Ports the flow of ``core_snpm_lmm.m``: read the long table (one row per trial), split metadata
(``lmm_meta_cols``) from channel columns, drop rows with a missing model variable (before
anything else), transform the channel values, select the montage's channels, fix the evaluable
channel set (complete across every row) once, then the TFCE loop and the cluster loop, each with its own permutations, and the
whole-head test (the effect refitted on the mean over the evaluable channels).

Parameter keys are MATLAB's: ``data_file`` (or ``data1_file``), ``data_sheet``, ``lmm_dv``,
``lmm_fixed``, ``lmm_random``, ``lmm_effect``, ``lmm_effect_type``, ``lmm_meta_cols``,
``lmm_perm`` (optional; derived from the effect type), ``lmm_subject`` (default ``Subject``),
``lmm_group`` (default ``group``), ``tail``, ``channels``, ``datatype``, ``permutations``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import lmm
from .io import tables
from .montage import ChannelSetup, setup_channels

COMPARISON = "mixedmodel"
CLUSTER_STATISTIC = ("cluster mass = sum of the Wald statistic (t² for a continuous effect, F "
                     "for a factor) over the cluster divided by the number of channels, i.e. the "
                     "mean Wald (Stephan et al. 2021)")


@dataclass
class LMMResults:
    tier = "lmm"
    params: dict
    spec: lmm.LMMSpec
    setup: ChannelSetup
    meta: pd.DataFrame
    power: np.ndarray
    evaluable: np.ndarray
    T: lmm.LMMTFCEResult
    clusters: lmm.LMMClusterResult
    global_stat: float
    global_p: float
    dropped: lmm.MetaDrop
    model_names: list[str]
    alpha: float
    notes: list[str] = field(default_factory=list)
    excluded: np.ndarray = field(default_factory=lambda: np.empty(0, int))
    excluded_reasons: list[str] = field(default_factory=list)
    within_ratio: np.ndarray | None = None   # per channel; NaN where not computed

    @property
    def n_trials(self) -> int:
        return int(self.power.shape[0])

    @property
    def n_subjects(self) -> int:
        return int(pd.unique(self.meta[self.spec.subject]).size)


def _g3(x: float) -> str:
    """MATLAB sprintf('%.3g'): two-digit exponent with sign, as C printf."""
    return f"{x:.3g}"


def _transform(a, datatype):
    from .pipeline import _transform as t
    return t(a, datatype)


def spec_from_params(params: dict) -> lmm.LMMSpec:
    missing = [k for k in ("lmm_dv", "lmm_fixed", "lmm_random", "lmm_effect", "lmm_effect_type")
               if not params.get(k)]
    if missing:
        raise ValueError(f"mixedmodel needs {', '.join(missing)}.")
    return lmm.LMMSpec(dv=str(params["lmm_dv"]), fixed=str(params["lmm_fixed"]),
                       random=str(params["lmm_random"]), effect=str(params["lmm_effect"]),
                       effect_type=str(params["lmm_effect_type"]),
                       subject=str(params.get("lmm_subject") or "Subject"),
                       group=str(params.get("lmm_group") or "group"),
                       perm=params.get("lmm_perm") or None,
                       categorical=None)


def prepare(params: dict):
    """(spec, meta, power, setup, evaluable, MetaDrop, notes), all decided before any permutation."""
    spec = spec_from_params(params)
    meta_cols = [str(c) for c in np.atleast_1d(params.get("lmm_meta_cols", []))]
    if not meta_cols:
        raise ValueError("params['lmm_meta_cols'] must list the non-channel columns (DV, Subject, "
                         "group, time, ...).")
    lmm.check_design(spec, params.get("tail", "both"), meta_cols)
    path = params.get("data_file") or params.get("data1_file")
    table = tables.read_table(path, params.get("data_sheet") or params.get("data1_sheet"))
    absent = [c for c in meta_cols if c not in table.columns]
    if absent:
        raise ValueError(f"lmm_meta_cols name column(s) not in the file: {', '.join(absent)}")
    channel_cols = [str(c) for c in table.columns if str(c) not in meta_cols]
    if not channel_cols:
        raise ValueError("No channel columns left after removing the meta columns.")
    meta = table[meta_cols].reset_index(drop=True)
    power = tables.numeric_matrix(table[channel_cols], "The data file")
    # Rows with a missing model variable go first, before the transform, the channel setup and
    # the evaluable mask, as in core_snpm_lmm.m (warnings snpm:lmmMetaNaNRows and
    # snpm:lmmSingleTrialSubject are raised by drop_missing_meta itself).
    meta, power, dropped = lmm.drop_missing_meta(meta, spec, power)
    notes: list[str] = list(dropped.notes)
    # within_subject shuffles the DV inside each subject: a DV constant within every subject
    # leaves nothing to permute. Labelling-independent, so decided once, here.
    if spec.perm == "within_subject":
        w_dv = float(lmm.within_ratio(meta[spec.dv].to_numpy(dtype=float),
                                      meta[spec.subject])[0])
        if w_dv <= lmm.WITHIN_MIN:
            raise lmm.LMMInputError(
                "dvConstantWithinSubject",
                f"The within_subject scheme shuffles '{spec.dv}' within each subject, but it is "
                f"constant within each subject (within-subject share of its variation "
                f"{_g3(w_dv)}, threshold 1e-06), so the within-subject shuffle has nothing to "
                "permute. Use group_label for a between-group effect, or analyse subject means.")
    power = _transform(power, params.get("datatype", "absolute"))
    setup = setup_channels(params.get("channels", "egi"), [power], channel_cols)
    power = np.array(setup.data[0], dtype=float)
    if power.shape[1] != setup.graph.n:
        raise ValueError(f"Channel count ({power.shape[1]}) does not match the neighbour matrix "
                         f"({setup.graph.n} rows).")
    power[~np.isfinite(power)] = np.nan
    evaluable = np.isfinite(power).all(axis=0)
    if not evaluable.any():
        raise lmm.LMMInputError(
            "noEvaluableChannels",
            f"No channel is complete across all {power.shape[0]} analysed rows, so no channel can "
            "be tested against a permutation null. Drop the incomplete subjects/trials instead of "
            "the channels.")
    n_out = int((~evaluable).sum())
    if n_out > 0.10 * evaluable.size:
        notes.append(f"{n_out} of {evaluable.size} channels ({100 * n_out / evaluable.size:.1f}%) "
                     "are incomplete and were dropped. Above about 10% of the montage, dropping "
                     "the few subjects or trials that carry the missingness usually costs less "
                     "information than dropping the channels; consider filtering the input table "
                     "instead.")
        warnings.warn(notes[-1], UserWarning, stacklevel=3)
    # Excluded channels, each with its reason, in channel order (MATLAB snpm_excluded_add: a
    # channel keeps its first reason).
    reasons = {int(i): lmm.MISSING for i in np.flatnonzero(~evaluable)}
    # Channels whose value is constant within each subject (within-subject share <= 1e-6),
    # grouped by subject only, so decided once and held for every permutation: as the response
    # (dv = POWER) there is nothing within subject to estimate; as the predictor under
    # within_subject its effect is between-subject only and the shuffle cannot move it.
    w_ch = np.full(evaluable.size, np.nan)
    w_ch[evaluable] = lmm.within_ratio(power[:, evaluable], meta[spec.subject])
    with np.errstate(invalid="ignore"):
        const = evaluable & (w_ch <= lmm.WITHIN_MIN)
    why = (lmm.CONST_RESPONSE if spec.dv == "POWER" else
           lmm.CONST_PREDICTOR if spec.perm == "within_subject" else None)
    if why is not None and const.any():
        for i in np.flatnonzero(const):
            reasons.setdefault(int(i), why)
        evaluable = evaluable & ~const
    if not evaluable.any():
        distinct = list(dict.fromkeys(reasons[i] for i in sorted(reasons)))
        raise lmm.LMMInputError(
            "noEvaluableChannels",
            f"No channel is left to test: {len(reasons)} of {evaluable.size} excluded "
            f"({'; '.join(distinct)}).")
    return spec, meta, power, setup, evaluable, dropped, notes, reasons, w_ch


def run_lmm(params: dict, rng: np.random.Generator | None = None, perms_tfce=None,
            perms_cluster=None, alpha: float = 0.05) -> LMMResults:
    """Run a mixed-model analysis end to end (no files are written here)."""
    from .pipeline import DEFAULT_PERMUTATIONS, TFCE_E, TFCE_H

    spec, meta, power, setup, ev, dropped, notes, reasons, w_ch = prepare(params)
    rng = rng if rng is not None else np.random.default_rng(params.get("seed"))
    n_perm = params.get("permutations", DEFAULT_PERMUTATIONS)
    tail = params.get("tail", "both")
    T = lmm.lmm_tfce(power, meta, spec, setup.graph, tail, alpha,
                     None if perms_tfce is not None else n_perm, TFCE_E, TFCE_H, ev, rng,
                     perms_tfce)
    C = lmm.lmm_cluster(power, meta, spec, setup.graph, alpha,
                        None if perms_cluster is not None else n_perm, ev, rng, perms_cluster)
    model = lmm.LMM(meta, spec)
    gstat, gp = global_test(meta, power, spec, ev)
    # observed fits caught by the per-labelling backstop: NaN in the map, listed
    for i in T.undefined_channels:
        reasons.setdefault(int(i), lmm.PERFECT_FIT)
    excluded = np.array(sorted(reasons), dtype=int)
    if excluded.size:
        notes.append(excluded_text(spec, setup.labels, excluded, [reasons[i] for i in excluded],
                                   ev.size))
    if T.n_degenerate:
        notes.append(f"{T.n_degenerate} of {T.n_perm} permutations produced no finite statistic "
                     "on any evaluable channel; those null entries were set to +Inf "
                     "(conservative).")
    return LMMResults(params, spec, setup, meta, power, ev, T, C, gstat, gp, dropped,
                      model.names, alpha, notes, excluded, [reasons[i] for i in excluded], w_ch)


def excluded_text(spec, labels, excluded, reasons, n_channels) -> str:
    """The results-text line of core_snpm_lmm.m (``lmm_excluded_text``)."""
    ctx = f"mixedmodel ({spec.effect})"
    n = len(excluded)
    if n == 0:
        return f"Channels excluded for missing data: none ({n_channels}/{n_channels} evaluable)"
    if all(r.startswith("missing data") for r in reasons):
        return (f"Channels excluded for missing data ({ctx}): {n} of {n_channels} — "
                f"{', '.join(labels[i] for i in excluded)}")
    parts = [f"{', '.join(labels[i] for i, ri in zip(excluded, reasons) if ri == r)} ({r})"
             for r in dict.fromkeys(reasons)]
    return f"Channels excluded ({ctx}): {n} of {n_channels} — {'; '.join(parts)}"


def global_test(meta, power, spec, evaluable) -> tuple[float, float]:
    """The effect refitted on the mean over the evaluable channels (descriptive, uncorrected)."""
    fit = lmm.LMM(meta, spec).fit(np.nanmean(power[:, evaluable], axis=1)[:, None])
    return float(fit.stat[0]), float(fit.p_model[0])
