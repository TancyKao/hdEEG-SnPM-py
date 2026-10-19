"""Circular (phase) tier: import, convention check and the statistics.

Ports ``snpm_circ_import.m``, ``snpm_circ_check_inversion.m``, ``snpm_circ_hotelling.m``, ``snpm_circ_hotelling_perm.m``, ``snpm_circ_watsons_u2.m``,
``watson_u2_pvalue.m``, ``snpm_circ_corr_anglinear.m``, ``snpm_circ_count_covariate.m``,
``snpm_circ_fl_context.m``, ``snpm_circ_linearise.m`` and the descriptive panel of
``core_snpm_circ.m`` (``circ_descriptive``, ``circ_mean_deg``). The permutation driver and the
file-to-results flow are in ``circ_pipeline.py``.

Every function takes angles in radians as a (subjects, channels) matrix with NaN for missing.
Properties that make the tier correct, as in MATLAB:

- The angle matrix is never permuted. Freedman-Lane permutes the stacked response
  ``[cos A, sin A]`` (``fl_context``) with ONE row index for all ``2 * n_ch`` columns
  (``permuted_embedding``), so cos and sin cannot desynchronise; the permuted response is off the
  unit circle and is scored by ``HotellingPerm``, not by ``hotelling``.
- The event-count covariate is per channel (``count_covariate``).
- The estimability gate (``min_per_group`` / ``min_n``) defaults to off inside every statistic:
  the caller computes it once from the observed labels and passes it as ``valid_channels`` to
  every permuted call.
- No whole-head test: ``descriptive`` is a no-inference panel.

Missing data inside ``hotelling`` and ``corr_anglinear`` is deleted per channel, so ``df2`` is a
per-channel vector, as in MATLAB.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .glm import FLContext

_EPS = np.finfo(float).eps
_REALMIN = np.finfo(float).tiny
TWO_PI = 2 * np.pi
U2_SERIES_FLOOR = 0.0065

DESCRIPTIVE_NOTE = (
    "Descriptive only. No whole-head inferential test is reported for the circular tier: an "
    "arithmetic channel mean is invalid for angles, and a circularly correct whole-head mean "
    "averages over a real 25-50 degree anterior-posterior phase gradient. Inference is the "
    "channel-wise permutation map above.")


class CircInputError(ValueError):
    """Invalid input to a circular statistic; ``code`` is the MATLAB identifier's suffix."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- import

CONVENTIONS = {"literature_uppeak0": 0.0, "yasa_uppeak0": 0.0, "luna_zerocross0": 90.0,
               "turtlewave_pre_v4": 180.0, "custom": None}
_CONVENTION_LIST = ("literature_uppeak0 | luna_zerocross0 | yasa_uppeak0 | turtlewave_pre_v4 | "
                    "custom")


def _mod(x, y):
    """MATLAB ``mod``: x - floor(x / y) * y."""
    return x - np.floor(x / y) * y


def import_angles(raw, units, convention, *, zero_offset_deg=None, resolution_gate="warn",
                  label="angles") -> tuple[np.ndarray, dict]:
    """Radians on [0, 2 pi) with zero at the slow-oscillation up-state peak; port of
    ``snpm_circ_import.m``.

    ``units`` ('rad'/'radians'/'deg'/'degrees') and ``convention`` are REQUIRED with no default:
    a wrong guess rotates every angle invisibly. Out-of-range values are an error, never a
    silent rescale. ``resolution_gate='error'`` (Watson U^2) refuses whole-degree or quantised
    input (``circResolutionTooCoarse``); ``'warn'`` (Hotelling, Tier 1) warns and proceeds.
    Returns the angles and the import record (``units``, ``convention``, ``rotation_deg``,
    ``resolution_deg``, ``quantised``, ``whole_degree``, ``n_finite``, ``label``).
    """
    gate = str(resolution_gate).lower()
    if gate not in ("warn", "error"):
        raise CircInputError("circResolutionGate", "opts.resolution_gate must be 'warn' or "
                             "'error'.")
    if not isinstance(units, str) or not units:
        raise CircInputError(
            "circUnitsRequired",
            "params.circ_units is REQUIRED for a circular analysis and has no default: it must "
            "be 'rad' or 'deg'. Reading degrees as radians scrambles every angle and no "
            "downstream check can recover it.")
    u = units.lower()
    if u in ("rad", "radians"):
        u = "rad"
    elif u in ("deg", "degrees"):
        u = "deg"
    else:
        raise CircInputError("circUnitsRequired",
                             f"params.circ_units must be 'rad' or 'deg'; got '{u}'.")

    raw = np.asarray(raw, dtype=float)
    fin = np.isfinite(raw)
    vals = raw[fin]
    if vals.size == 0:
        raise CircInputError("circNoAngles", f"{label} contains no finite angles.")
    mx = float(np.max(np.abs(vals)))
    if u == "rad":
        if mx > 2 * np.pi + 1e-9:
            raise CircInputError(
                "circUnitsOutOfRange",
                f"{label} was declared as RADIANS but contains |angle| = {mx:.3f} > 2*pi. This "
                "is almost certainly a degree export. Fix the declaration (params.circ_units = "
                "'deg'); the units are never converted silently.")
        adeg = raw * 180 / np.pi
    else:
        if mx > 360 + 1e-9:
            raise CircInputError(
                "circUnitsOutOfRange",
                f"{label} was declared as DEGREES but contains |angle| = {mx:.3f} > 360. Check "
                "the export; the units are never converted silently.")
        adeg = raw

    dvals = adeg[fin]
    res = float(np.max(np.abs(dvals - np.round(dvals))))
    whole, quant = res < 1e-6, res < 0.05
    if quant:
        what = (f"whole-degree (max|deg - round(deg)| = {res:.3g})" if whole
                else f"quantised to ~{res:.3g} deg")
        if gate == "error":
            raise CircInputError(
                "circResolutionTooCoarse",
                f"{label} is {what}, which Watson's U^2 cannot use. U^2 is built on the "
                "empirical distribution functions, so tied angles are structural, not cosmetic: "
                "whole-degree input moves the uncorrected p by up to 0.084 and drops the "
                "topographic rank correlation against the full-precision map to 0.958. Three "
                "remedies, any one is enough: (1) export preferred_phase_rad -- TurtleWave "
                "stores it as a float, so this costs nothing; (2) re-derive the circular mean "
                "at full precision instead of reading Luna's whole-degree COUPL_ANGLE; (3) use "
                "the Hotelling path (comparison = 'circ_phase_group'), which is the default and "
                "is unaffected because it is a smooth function of the angle with no ranks "
                "anywhere.")
        warnings.warn(
            f"{label} is {what}. The Hotelling / Tier-1 paths are smooth functions of the "
            "angle, so a <=0.5 deg perturbation against a 15-50 deg between-subject circular SD "
            "costs a fraction of a percent -- proceeding. Watson's U^2 would REJECT this input.",
            stacklevel=2)

    if not isinstance(convention, str) or not convention:
        raise CircInputError(
            "circConventionRequired",
            "params.circ_convention is REQUIRED and has no default. Choose one of: "
            f"{_CONVENTION_LIST} (with params.circ_zero_offset_deg).")
    conv = convention.lower()
    if conv not in CONVENTIONS:
        raise CircInputError("circConventionRequired",
                             f"Unknown params.circ_convention '{conv}'. Choose one of: "
                             f"{_CONVENTION_LIST}.")
    rot = CONVENTIONS[conv]
    if conv == "custom":
        ok = isinstance(zero_offset_deg, (int, float, np.integer, np.floating)) and \
            not isinstance(zero_offset_deg, bool) and np.isfinite(zero_offset_deg)
        if not ok:
            raise CircInputError(
                "circConventionRequired",
                "convention 'custom' needs a finite scalar params.circ_zero_offset_deg (degrees "
                "to add so that 0 lands on the slow-oscillation up-state peak).")
        rot = float(zero_offset_deg)

    A = _mod((adeg + rot) * np.pi / 180, TWO_PI)
    A[~fin] = np.nan
    info = {"units": u, "convention": conv, "rotation_deg": float(_mod(rot, 360.0)),
            "resolution_deg": res, "quantised": bool(quant), "whole_degree": bool(whole),
            "n_finite": int(vals.size), "label": label}
    return A, info


def check_inversion(A, gate, x_coords, imp: dict) -> dict:
    """Hard stop on a 180-degree phase-convention error; port of ``snpm_circ_check_inversion.m``.

    Pools both groups (label-invariant, so it cannot contaminate the inference) over the
    channels in ``gate``, restricted to the anterior third by chanlocs X (``x_coords``, +X toward
    the nose) when that leaves at least 3. A pooled mean direction within 70 degrees of exact
    antiphase raises ``circPhaseConventionInverted``; a deviation above 70 degrees warns.
    Returns ``channels`` (0-based), ``n_angles``, ``mean_deg``, ``deviation_deg``, ``verdict``.
    """
    A = _as_angles(A)
    info = {"channels": np.zeros(0, dtype=int), "n_angles": 0, "mean_deg": np.nan,
            "deviation_deg": np.nan, "verdict": "skipped"}
    ch = np.flatnonzero(np.asarray(gate, dtype=bool).ravel())
    if ch.size == 0:
        return info
    if x_coords is not None:
        x = np.asarray(x_coords, dtype=float).ravel()[ch]
        if np.isfinite(x).any():
            thr = np.quantile(x[np.isfinite(x)], 2 / 3, method="hazen")   # MATLAB quantile
            with np.errstate(invalid="ignore"):
                sel = ch[x >= thr]
            if sel.size >= 3:
                ch = sel
    a = A[:, ch]
    a = a[np.isfinite(a)]
    info["channels"] = ch
    info["n_angles"] = int(a.size)
    if a.size == 0:
        return info
    mu = float(_mod(np.degrees(np.angle(np.exp(1j * a).sum())), 360.0))
    dev = float(_mod(mu + 180, 360.0) - 180)
    info.update(mean_deg=mu, deviation_deg=dev, verdict="pass")
    if abs(dev) >= 110:
        raise CircInputError(
            "circPhaseConventionInverted",
            f"The pooled grand mean preferred phase over frontal channels is {mu:.1f} deg "
            f"({dev:+.1f} deg from the slow-oscillation up-state peak), i.e. inside the inverted "
            "window [110, 250] deg -- within 70 deg of exactly antiphase. Coupling to the DOWN "
            "state at this magnitude is not a finding, it is the known TurtleWave pre-v4.0 "
            f"phase inversion. The declared convention was '{imp['convention']}' (rotation "
            f"{imp['rotation_deg']:+.1f} deg applied). Repaired it would read "
            f"{_mod(mu + 180, 360.0):.1f} deg. Two remedies: (1) re-export from TurtleWave v4.0 "
            "or later, which fixes the sign at source; or (2) if you are certain the export "
            "predates v4.0, declare it -- params.circ_convention = 'turtlewave_pre_v4' -- and "
            "this check will re-run after the 180 deg rotation and must then pass.")
    if abs(dev) > 70:
        info["verdict"] = "suspect"
        warnings.warn(f"Pooled grand mean preferred phase is {dev:+.1f} deg from the up-state "
                      f"peak (convention '{imp['convention']}'). That is a large deviation; "
                      "confirm the zero convention of your export before reading the result.",
                      stacklevel=2)
    return info


# --------------------------------------------------------------------------- helpers

def _as_angles(A) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    return A.reshape(-1, 1) if A.ndim == 1 else A


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, str) and v == ""


def two_level_indicator(g) -> tuple[np.ndarray, list[str]]:
    """Map a two-level grouping variable to a float 0/1 column (NaN where missing).

    As MATLAB: logical keeps False=0/True=1; numeric levels are sorted and the larger is 1;
    text and categorical levels are sorted (categorical: category order) and the second is 1.
    The statistics are symmetric in the two groups, so the choice only affects labelling.
    """
    if isinstance(g, pd.Series):
        g = g.array
    if isinstance(g, pd.Categorical):
        vals = np.asarray(g, dtype=object)
        present = [c for c in g.categories if (vals == c).any()]     # removecats
        if len(present) != 2:
            raise CircInputError("twoLevels", f"g must have exactly two distinct levels "
                                 f"(found {len(present)}).")
        gi = np.array([np.nan if _is_missing(v) else float(v == present[1]) for v in vals])
        return gi, [str(c) for c in present]
    g = np.asarray(g).ravel()
    if g.dtype == bool:
        return g.astype(float), ["false", "true"]
    if np.issubdtype(g.dtype, np.number):
        g = g.astype(float)
        u = np.unique(g[~np.isnan(g)])
        if u.size != 2:
            raise CircInputError("twoLevels", f"g must have exactly two distinct levels "
                                 f"(found {u.size}).")
        gi = np.full(g.shape, np.nan)
        gi[g == u[0]] = 0.0
        gi[g == u[1]] = 1.0
        return gi, [f"{u[0]:.5g}", f"{u[1]:.5g}"]
    vals = [None if _is_missing(v) else str(v) for v in g.tolist()]
    cats = sorted({v for v in vals if v is not None})
    if len(cats) != 2:
        raise CircInputError("twoLevels", f"g must have exactly two distinct levels "
                             f"(found {len(cats)}).")
    gi = np.array([np.nan if v is None else float(v == cats[1]) for v in vals])
    return gi, cats


def _apply_gates(n1, n2, reason, min_per_group, valid_channels, n_ch, *arrays):
    """MATLAB's two estimability gates; NaN the arrays in place where a gate fails."""
    if min_per_group > 0:
        bad = (n1 < min_per_group) | (n2 < min_per_group)
        for ch in np.flatnonzero(bad):
            if not reason[ch]:
                reason[ch] = f"below min_per_group ({n1[ch]}/{n2[ch]})"
        for a in arrays:
            a[bad] = np.nan
    _apply_mask(valid_channels, n_ch, reason, *arrays)


def _apply_mask(valid_channels, n_ch, reason, *arrays):
    if valid_channels is None:
        return
    valid = np.asarray(valid_channels, dtype=bool).ravel()
    if valid.size != n_ch:
        raise CircInputError("maskSize", f"valid_channels has {valid.size} entries, "
                             f"expected {n_ch}.")
    for ch in np.flatnonzero(~valid):
        if not reason[ch]:
            reason[ch] = "excluded by valid_channels"
    for a in arrays:
        a[~valid] = np.nan


def _quad(dc, ds, scc, sss, scs, det):
    """d' S^-1 d for a 2x2 S, elementwise over channels."""
    return (dc ** 2 * sss - 2 * dc * ds * scs + ds ** 2 * scc) / det


# --------------------------------------------------------------------------- Hotelling

@dataclass
class HotellingResult:
    F: np.ndarray          # (n_ch,) Hotelling F, NaN where not estimable
    p: np.ndarray          # (n_ch,) parametric upper-tail p from F(2, df2)
    df1: int
    df2: np.ndarray        # (n_ch,) per-channel nu - 1 = N_ch - q - 3
    T2: np.ndarray
    n1: np.ndarray         # retained subjects in level 2 (indicator 1)
    n2: np.ndarray         # retained subjects in level 1 (indicator 0)
    n: np.ndarray
    dcos: np.ndarray       # adjusted group difference in the embedding, level 2 - level 1
    dsin: np.ndarray
    levels: list[str]
    reason: list[str]


def _hotelling_block(zc, zs, X, det_floor):
    m = zc.shape[1]
    nan = np.full(m, np.nan)
    n_obs = X.shape[0]
    rank = np.linalg.matrix_rank(X)
    nu = n_obs - rank
    if rank < X.shape[1]:
        return nan, nan.copy(), nu, nan.copy(), nan.copy(), \
            ["design rank deficient on retained rows"] * m
    if nu < 2:
        return nan, nan.copy(), nu, nan.copy(), nan.copy(), \
            [f"too few observations (N={n_obs}, rank(X)={rank})"] * m
    xtx_inv = np.linalg.solve(X.T @ X, np.eye(X.shape[1]))
    bc = xtx_inv @ (X.T @ zc)
    bs = xtx_inv @ (X.T @ zs)
    rc = zc - X @ bc
    rs = zs - X @ bs
    scc = (rc ** 2).sum(axis=0) / nu
    sss = (rs ** 2).sum(axis=0) / nu
    scs = (rc * rs).sum(axis=0) / nu
    dc, ds = bc[-1], bs[-1]                       # the contrast selects the g column
    v = xtx_inv[-1, -1]
    det = scc * sss - scs ** 2
    sing = ~(det > det_floor)
    with np.errstate(invalid="ignore", divide="ignore"):
        t2 = _quad(dc, ds, scc, sss, scs, det) / v
    f = t2 * (nu - 1) / (2 * nu)
    f[sing] = np.nan
    t2[sing] = np.nan
    why = ["residual covariance singular (all angles identical?)" if s else "" for s in sing]
    return f, t2, nu, dc.copy(), ds.copy(), why


def hotelling(A, g, cnuis=None, *, min_per_group: int = 0, valid_channels=None,
              det_floor: float = 1e-12) -> HotellingResult:
    """Two-sample Hotelling T^2 on the (cos, sin) embedding per channel, covariate-adjusted.

    Design ``X = [1, cnuis, g]``, response ``Z = [cos a, sin a]``, contrast on ``g``:
    ``T2 = d' S^-1 d / (c'(X'X)^-1 c)``, ``F = T2 (nu - 1) / (2 nu) ~ F(2, nu - 1)``. At no
    covariates this is the classical pooled-covariance two-sample form. Rows with NaN in
    ``cnuis`` or ``g`` are dropped at every channel; NaN angles are deleted per channel.
    """
    A = _as_angles(A)
    n_subj, n_ch = A.shape
    gi, levels = two_level_indicator(g)
    if gi.size != n_subj:
        raise CircInputError("groupSize", f"g has {gi.size} entries but A has {n_subj} rows.")
    cn = np.zeros((n_subj, 0)) if cnuis is None else np.asarray(cnuis, dtype=float)
    if cn.ndim == 1:
        cn = cn.reshape(-1, 1)
    if cn.shape[0] != n_subj:
        raise CircInputError("covSize", f"Cnuis has {cn.shape[0]} rows but A has {n_subj} rows.")
    X = np.column_stack([np.ones(n_subj), cn, gi])

    F = np.full(n_ch, np.nan)
    T2 = np.full(n_ch, np.nan)
    df2 = np.full(n_ch, np.nan)
    dcos = np.full(n_ch, np.nan)
    dsin = np.full(n_ch, np.nan)
    reason = [""] * n_ch

    row_ok = np.isfinite(X).all(axis=1)
    ok_a = ~np.isnan(A) & row_ok[:, None]
    n1 = (ok_a & (gi == 1)[:, None]).sum(axis=0)
    n2 = (ok_a & (gi == 0)[:, None]).sum(axis=0)
    nn = n1 + n2
    zc, zs = np.cos(A), np.sin(A)

    is_full = (ok_a == row_ok[:, None]).all(axis=0) & (nn > 0)
    full = np.flatnonzero(is_full)
    if full.size:
        f, t2, nu, dc, ds, why = _hotelling_block(zc[row_ok][:, full], zs[row_ok][:, full],
                                                  X[row_ok], det_floor)
        F[full], T2[full], df2[full], dcos[full], dsin[full] = f, t2, nu - 1, dc, ds
        for k, ch in enumerate(full):
            reason[ch] = why[k]
    for ch in np.flatnonzero(~is_full):
        rows = ok_a[:, ch]
        if not rows.any():
            reason[ch] = "no observations"
            continue
        f, t2, nu, dc, ds, why = _hotelling_block(zc[rows, ch:ch + 1], zs[rows, ch:ch + 1],
                                                  X[rows], det_floor)
        F[ch], T2[ch], df2[ch], dcos[ch], dsin[ch] = f[0], t2[0], nu - 1, dc[0], ds[0]
        reason[ch] = why[0]

    _apply_gates(n1, n2, reason, min_per_group, valid_channels, n_ch, F, T2, df2)
    ok = np.isfinite(F) & np.isfinite(df2) & (df2 > 0)
    p = np.full(n_ch, np.nan)
    p[ok] = stats.f.sf(F[ok], 2, df2[ok])
    F[~ok] = np.nan
    T2[~ok] = np.nan
    df2[~ok] = np.nan
    return HotellingResult(F, p, 2, df2, T2, n1, n2, nn, dcos, dsin, levels, reason)


class HotellingPerm:
    """Permutation-side Hotelling evaluator on a (cos, sin) response with a per-channel covariate.

    Port of ``snpm_circ_hotelling_perm.m``. Under Freedman-Lane the permuted response is
    ``fit + resid[pe]``, which is off the unit circle, so it is scored on the embedding directly.
    ``(X'X)^-1`` per covariate channel is built once here. Two regimes: where ``usecov`` is true
    the design is ``[1, cov_ch, g]`` on complete columns (the caller guarantees completeness);
    elsewhere it is the classical two-sample form with per-channel deletion. Channels outside
    ``valid`` are NaN in every call, so the observed and every permuted map share one channel set.
    """

    def __init__(self, gi, cov, usecov, valid):
        self.gi = np.asarray(gi, dtype=float).ravel()
        n_subj = self.gi.size
        usecov = np.asarray(usecov, dtype=bool).ravel()
        n_ch = usecov.size
        cov = np.zeros((n_subj, n_ch)) if cov is None or np.size(cov) == 0 else \
            np.asarray(cov, dtype=float)
        self.valid = np.asarray(valid, dtype=bool).ravel().copy()
        self.n_ch = n_ch
        cov_ch = np.flatnonzero(self.valid & usecov)
        self.pln_ch = np.flatnonzero(self.valid & ~usecov)
        gi = self.gi
        s1, sg, sgg = n_subj, gi.sum(), (gi ** 2).sum()
        keep, inv = [], []
        self.vgg = np.zeros(n_ch)
        for ch in cov_ch:
            c = cov[:, ch]
            M = np.array([[s1, c.sum(), sg], [c.sum(), c @ c, c @ gi], [sg, gi @ c, sgg]])
            with np.errstate(all="ignore"):
                try:
                    Mi_full = np.linalg.inv(M)
                    rc = 1.0 / (np.abs(M).sum(axis=0).max() * np.abs(Mi_full).sum(axis=0).max())
                except np.linalg.LinAlgError:
                    rc = 0.0
            if not rc >= 1e-12:                  # MATLAB: rcond(M) < 1e-12 (NaN skips too)
                continue
            Mi = np.linalg.solve(M, np.eye(3))
            keep.append(ch)
            inv.append(Mi)
            self.vgg[ch] = Mi[2, 2]
        self.cov_ch = np.array(keep, dtype=int)
        self.xtx_inv = np.array(inv).reshape(-1, 3, 3)
        self.cov_c = cov[:, self.cov_ch]
        self.nu_cov = n_subj - 3
        self.nu = np.full(n_ch, np.nan)
        self.nu[self.cov_ch] = self.nu_cov

    def evaluate(self, yc, ys) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(F, p, df2) for a cos/sin response pair, each (subjects, channels)."""
        yc = np.asarray(yc, dtype=float)
        ys = np.asarray(ys, dtype=float)
        n_ch, gi = self.n_ch, self.gi
        F = np.full(n_ch, np.nan)
        df2 = np.full(n_ch, np.nan)
        if self.cov_ch.size and self.nu_cov >= 2:
            ch, cv, nu = self.cov_ch, self.cov_c, self.nu_cov
            out = []
            for y in (yc[:, ch], ys[:, ch]):
                rhs = np.stack([y.sum(axis=0), (cv * y).sum(axis=0), gi @ y], axis=1)  # (m, 3)
                B = np.einsum("mij,mj->im", self.xtx_inv, rhs)                          # (3, m)
                r = y - (B[0] + cv * B[1] + gi[:, None] * B[2])
                out.append((B[2], r))
            (dc, rc), (ds, rs) = out
            scc = (rc ** 2).sum(axis=0) / nu
            sss = (rs ** 2).sum(axis=0) / nu
            scs = (rc * rs).sum(axis=0) / nu
            det = scc * sss - scs ** 2
            with np.errstate(invalid="ignore", divide="ignore"):
                t2 = _quad(dc, ds, scc, sss, scs, det) / self.vgg[ch]
            f = t2 * (nu - 1) / (2 * nu)
            f[~(det > 1e-12)] = np.nan
            F[ch] = f
            df2[ch] = nu - 1
        if self.pln_ch.size:
            ch = self.pln_ch
            c, s = yc[:, ch], ys[:, ch]
            ok = np.isfinite(c) & np.isfinite(s)
            m1 = ok & (gi == 1)[:, None]
            m2 = ok & (gi == 0)[:, None]
            n1, n2 = m1.sum(axis=0), m2.sum(axis=0)
            N = n1 + n2
            nu_p = N - 2
            c0 = np.where(ok, c, 0.0)
            s0 = np.where(ok, s, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                c1 = (c0 * m1).sum(axis=0) / np.maximum(n1, 1)
                c2 = (c0 * m2).sum(axis=0) / np.maximum(n2, 1)
                s1 = (s0 * m1).sum(axis=0) / np.maximum(n1, 1)
                s2 = (s0 * m2).sum(axis=0) / np.maximum(n2, 1)
                rcm = np.where(ok, c0 - (m1 * c1 + m2 * c2), 0.0)
                rsm = np.where(ok, s0 - (m1 * s1 + m2 * s2), 0.0)
                scc = (rcm ** 2).sum(axis=0) / nu_p
                sss = (rsm ** 2).sum(axis=0) / nu_p
                scs = (rcm * rsm).sum(axis=0) / nu_p
                dc, ds = c1 - c2, s1 - s2
                det = scc * sss - scs ** 2
                t2 = _quad(dc, ds, scc, sss, scs, det) * (n1 * n2 / N)
                f = t2 * (nu_p - 1) / (2 * nu_p)
            bad = ~(det > 1e-12) | (n1 < 1) | (n2 < 1) | (nu_p < 2)
            f[bad] = np.nan
            F[ch] = f
            d = (nu_p - 1).astype(float)
            d[bad] = np.nan
            df2[ch] = d
        good = np.isfinite(F) & np.isfinite(df2) & (df2 > 0)
        p = np.full(n_ch, np.nan)
        p[good] = stats.f.sf(F[good], 2, df2[good])
        F[~good] = np.nan
        df2[~good] = np.nan
        return F, p, df2


# --------------------------------------------------------------------------- Watson U^2

@dataclass
class WatsonResult:
    U2: np.ndarray
    p: np.ndarray          # asymptotic, from watson_u2_pvalue (cluster forming, description)
    n1: np.ndarray
    n2: np.ndarray
    n: np.ndarray
    reason: list[str]


def watson_u2(A, g, *, min_per_group: int = 0, valid_channels=None) -> WatsonResult:
    """Watson's two-sample U^2 per channel, ECDF form with ties weighted by multiplicity.

    Over the distinct pooled values v_k with multiplicity m_k and d_k = i_k/n1 - j_k/n2,
    ``U2 = (n1 n2 / N^2) [sum m_k d_k^2 - (sum m_k d_k)^2 / N]``. The result does not depend on
    how tied observations are ordered. Angles are wrapped to [0, 2 pi); NaN sorts last.
    """
    A = _as_angles(A)
    n_subj, n_ch = A.shape
    gi, _ = two_level_indicator(g)
    if gi.size != n_subj:
        raise CircInputError("groupSize", f"g has {gi.size} entries but A has {n_subj} rows.")
    Aw = np.mod(A, TWO_PI)
    Aw[~np.isfinite(A)] = np.nan
    Aw[~np.isfinite(gi), :] = np.nan
    order = np.argsort(Aw, axis=0, kind="stable")             # NaN last, as MATLAB
    As = np.take_along_axis(Aw, order, axis=0)
    isv = ~np.isnan(As)
    if n_subj > 1:
        is_end = np.vstack([As[:-1] != As[1:], np.ones((1, n_ch), bool)])
    else:
        is_end = np.ones((n_subj, n_ch), bool)
    in1 = (gi == 1)[order] & isv
    in2 = (gi == 0)[order] & isv
    ik = np.cumsum(in1, axis=0)
    jk = np.cumsum(in2, axis=0)
    n1 = ik[-1] if n_subj else np.zeros(n_ch, int)
    n2 = jk[-1] if n_subj else np.zeros(n_ch, int)
    nn = n1 + n2
    usable = (n1 > 0) & (n2 > 0) & (nn > 2)

    U2 = np.full(n_ch, np.nan)
    reason = ["" if u else f"degenerate split (n1={n1[ch]}, n2={n2[ch]})"
              for ch, u in enumerate(usable)]
    if usable.any():
        u = usable
        d = ik[:, u] / n1[u] - jk[:, u] / n2[u]
        eu = is_end[:, u]
        for k in range(n_subj - 2, -1, -1):            # value at the END of each tie block
            carry = ~eu[k]
            d[k, carry] = d[k + 1, carry]
        sd = d.sum(axis=0)
        sd2 = (d ** 2).sum(axis=0)
        U2[u] = (n1[u] * n2[u] / nn[u] ** 2) * (sd2 - sd ** 2 / nn[u])

    _apply_gates(n1, n2, reason, min_per_group, valid_channels, n_ch, U2)
    return WatsonResult(U2, watson_u2_pvalue(U2), n1, n2, nn, reason)


def watson_u2_pvalue(u) -> np.ndarray:
    """Asymptotic upper-tail p of Watson's U^2: ``2 sum_{k=1}^{12} (-1)^(k-1) exp(-2 k^2 pi^2 u)``.

    p = 1 at or below 0.0065 (where the truncated series stops converging), clamped to [0, 1],
    NaN preserved. Reproduces the five published critical values to better than 5e-4.
    """
    u = np.asarray(u, dtype=float)
    p = np.full(u.shape, np.nan)
    finite = np.isfinite(u)
    if not finite.any():
        return p
    uu = u[finite]
    k = np.arange(1, 13, dtype=float)[:, None]
    sgn = (-1.0) ** (k - 1)
    series = 2 * (sgn * np.exp(-2 * k ** 2 * np.pi ** 2 * uu[None, :])).sum(axis=0)
    series[uu <= U2_SERIES_FLOOR] = 1.0
    p[finite] = np.clip(series, 0.0, 1.0)
    return p


# --------------------------------------------------------------------------- circular-linear

@dataclass
class AngLinearResult:
    F: np.ndarray          # model F for the two embedding columns
    p: np.ndarray
    df1: int
    df2: np.ndarray        # (n_ch,) n_ch - 3 - q
    rho: np.ndarray        # raw covariate-free circular-linear r (descriptive, non-directional)
    n: np.ndarray
    R2: np.ndarray         # partial R^2 of the embedding after the covariates
    reason: list[str]


def _anglinear_block(y, Z, zc, zs, q, tol):
    m = zc.shape[1]
    F = np.full(m, np.nan)
    rho = np.full(m, np.nan)
    R2 = np.full(m, np.nan)
    n_obs = y.size
    rank = np.linalg.matrix_rank(Z)
    df2 = n_obs - rank - 2
    if rank < Z.shape[1]:
        return F, np.nan, rho, R2, ["nuisance design rank deficient on retained rows"] * m
    if df2 < 1:
        return F, np.nan, rho, R2, [f"too few observations (n={n_obs}, q={q})"] * m

    ztz_inv = np.linalg.solve(Z.T @ Z, np.eye(Z.shape[1]))
    yr = y - Z @ (ztz_inv @ (Z.T @ y))
    cr = zc - Z @ (ztz_inv @ (Z.T @ zc))
    sr = zs - Z @ (ztz_inv @ (Z.T @ zs))
    rss_red = (yr ** 2).sum()
    a11 = (cr ** 2).sum(axis=0)
    a22 = (sr ** 2).sum(axis=0)
    a12 = (cr * sr).sum(axis=0)
    b1 = yr @ cr
    b2 = yr @ sr
    det = a11 * a22 - a12 ** 2
    degen = ~(det > tol * np.maximum(a11 * a22, _EPS))
    with np.errstate(invalid="ignore", divide="ignore"):
        extra = np.fmax(_quad(b1, b2, a11, a22, a12, det), 0.0)   # MATLAB max ignores NaN
        rss_full = rss_red - extra
        flat = not rss_red > 0
        sat = ~(rss_full > tol * max(rss_red, _EPS))
        F = (extra / 2) / (rss_full / df2)
        R2 = extra / rss_red
    bad = degen | flat | sat
    F[bad] = np.nan
    R2[bad] = np.nan
    why = []
    for k in range(m):
        if degen[k]:
            why.append("angle embedding degenerate (all angles identical?)")
        elif flat:
            why.append("response constant on retained rows")
        elif sat[k]:
            why.append("residual sum of squares numerically zero")
        else:
            why.append("")

    # descriptive raw rho: covariate-free R^2 of y on [1, cos, sin]
    y0 = y - y.mean()
    c0 = zc - zc.mean(axis=0)
    s0 = zs - zs.mean(axis=0)
    A11 = (c0 ** 2).sum(axis=0)
    A22 = (s0 ** 2).sum(axis=0)
    A12 = (c0 * s0).sum(axis=0)
    B1 = y0 @ c0
    B2 = y0 @ s0
    det0 = A11 * A22 - A12 ** 2
    tss = (y0 ** 2).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        r2raw = _quad(B1, B2, A11, A22, A12, det0) / max(tss, _EPS)
    rho = np.sqrt(np.fmin(np.fmax(r2raw, 0.0), 1.0))
    rho[~(det0 > tol * np.maximum(A11 * A22, _EPS)) | (tss <= 0)] = np.nan
    return F, df2, rho, R2, why


def corr_anglinear(y, A, cnuis=None, *, min_n: int = 0, valid_channels=None,
                   tol: float = 1e-12) -> AngLinearResult:
    """Circular-linear association per channel as the partial F of ``[cos a, sin a]`` in
    ``y ~ 1 + cnuis + cos a + sin a``, F(2, n - 3 - q). ``rho`` is the covariate-free
    circular-linear correlation (CircStat ``circ_corrcl``), descriptive only.

    Permute ``y``, never ``A`` (one index for every channel).
    """
    A = _as_angles(A)
    y = np.asarray(y, dtype=float).ravel()
    n_subj, n_ch = A.shape
    if y.size != n_subj:
        raise CircInputError("respSize", f"y has {y.size} entries but A has {n_subj} rows.")
    cn = np.zeros((n_subj, 0)) if cnuis is None else np.asarray(cnuis, dtype=float)
    if cn.ndim == 1:
        cn = cn.reshape(-1, 1)
    if cn.shape[0] != n_subj:
        raise CircInputError("covSize", f"Cnuis has {cn.shape[0]} rows but A has {n_subj} rows.")
    q = cn.shape[1]
    Z = np.column_stack([np.ones(n_subj), cn])
    row_ok = np.isfinite(y) & np.isfinite(Z).all(axis=1)
    zc, zs = np.cos(A), np.sin(A)
    ok_a = ~np.isnan(A) & row_ok[:, None]
    nn = ok_a.sum(axis=0)

    F = np.full(n_ch, np.nan)
    df2 = np.full(n_ch, np.nan)
    rho = np.full(n_ch, np.nan)
    R2 = np.full(n_ch, np.nan)
    reason = [""] * n_ch

    is_full = (ok_a == row_ok[:, None]).all(axis=0) & (nn > 0)
    full = np.flatnonzero(is_full)
    if full.size:
        f, d, r, r2, why = _anglinear_block(y[row_ok], Z[row_ok], zc[row_ok][:, full],
                                            zs[row_ok][:, full], q, tol)
        F[full], df2[full], rho[full], R2[full] = f, d, r, r2
        for k, ch in enumerate(full):
            reason[ch] = why[k]
    for ch in np.flatnonzero(~is_full):
        rows = ok_a[:, ch]
        if not rows.any():
            reason[ch] = "no observations"
            continue
        f, d, r, r2, why = _anglinear_block(y[rows], Z[rows], zc[rows, ch:ch + 1],
                                            zs[rows, ch:ch + 1], q, tol)
        F[ch], df2[ch], rho[ch], R2[ch] = f[0], d, r[0], r2[0]
        reason[ch] = why[0]

    if min_n > 0:
        bad = nn < min_n
        for ch in np.flatnonzero(bad):
            if not reason[ch]:
                reason[ch] = f"below min_n (n={nn[ch]})"
        F[bad] = np.nan
        df2[bad] = np.nan
    _apply_mask(valid_channels, n_ch, reason, F, df2)
    ok = np.isfinite(F) & np.isfinite(df2) & (df2 > 0)
    p = np.full(n_ch, np.nan)
    p[ok] = stats.f.sf(F[ok], 2, df2[ok])
    F[~ok] = np.nan
    df2[~ok] = np.nan
    return AngLinearResult(F, p, 2, df2, rho, nn, R2, reason)


# --------------------------------------------------------------------------- count covariate

@dataclass
class CountCovariate:
    A: np.ndarray            # angles, NaN where a cell had an angle but no usable count
    logC: np.ndarray         # (n_subj, n_ch) log(count), NaN where unusable
    usecov: np.ndarray       # (n_ch,) covariate enters the design here
    collinear: np.ndarray    # (n_ch,) not estimable; the caller must exclude these channels
    report: dict = field(default_factory=dict)


def _point_biserial(X, gi):
    """Per-column ``corrcoef(v, g)`` over the finite rows, as MATLAB computes it: centre on a
    sequentially summed mean, ``cov ./ d ./ d'``, clip to [-1, 1]. A constant column then gives
    MATLAB's rounding-level value (0 or about 1e-17), not NaN, which enters the reported median."""
    r = np.full(X.shape[1], np.nan)
    for k in range(X.shape[1]):
        v = X[:, k]
        ok = np.isfinite(v)
        if ok.sum() < 3 or np.unique(gi[ok]).size < 2:
            continue
        n = int(ok.sum())
        a = v[ok] - np.cumsum(v[ok])[-1] / n
        b = gi[ok] - np.cumsum(gi[ok])[-1] / n
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (a @ b) / (n - 1) / np.sqrt((a @ a) / (n - 1)) / np.sqrt((b @ b) / (n - 1))
        r[k] = np.sign(c) if abs(c) > 1 else c
    return r


def _nanstd1(X):
    """MATLAB std(X, 0, 1, 'omitnan'): one finite value gives 0, none gives NaN."""
    ok = np.isfinite(X)
    n = ok.sum(axis=0)
    mu = np.where(ok, X, 0.0).sum(axis=0) / np.maximum(n, 1)
    ss = (np.where(ok, X - mu, 0.0) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(ss / np.maximum(n - 1, 1))
    sd[n == 0] = np.nan
    return sd


def count_covariate(A, gi, counts, *, mode: str = "per_channel", collinearity_max: float = 0.8,
                    labels=None) -> CountCovariate:
    """Validate and condition the per-channel log(event count) covariate (two-group tier).

    Port of ``snpm_circ_count_covariate.m``. ``counts`` is already aligned to ``A`` and ``gi``.
    A cell with an angle but no usable count (missing or zero) loses its angle; above 5% of angle
    cells that is an error. ``usecov`` is false where sd(log count) < 0.05; channels where log
    count is disjoint by group or |point-biserial| > ``collinearity_max`` are ``collinear``
    (above 25% of channels that is an error). ``report['diagnostic']`` is the confound text
    MATLAB prints before any circular result.
    """
    A = _as_angles(A).copy()
    gi = np.asarray(gi, dtype=float).ravel()
    C = np.asarray(counts, dtype=float)
    n_ch = A.shape[1]
    labels = [f"Ch{k + 1:03d}" for k in range(n_ch)] if labels is None else list(labels)
    if C.shape != A.shape:
        raise CircInputError("circCountsShape", f"Count matrix is {C.shape[0]}x{C.shape[1]} but "
                             f"the angle matrix is {A.shape[0]}x{A.shape[1]}.")
    with np.errstate(invalid="ignore"):
        bad = np.isfinite(C) & ((C < 0) | (np.abs(C - np.round(C)) > 1e-9))
    if bad.any():
        cols, rows = np.nonzero(bad.T)               # MATLAB find: column-major order
        items = [f"row {r + 1}, {labels[c]} = {C[r, c]:g}" for r, c in zip(rows[:5], cols[:5])]
        raise CircInputError("circCountsInvalid", "Event counts must be non-negative integers. "
                             f"Offending entries (first {len(items)}): {'; '.join(items)}.")

    have = np.isfinite(A)
    with np.errstate(invalid="ignore"):
        usable_count = np.isfinite(C) & (C > 0)
    no_count = have & ~usable_count
    n_a, n_miss = int(have.sum()), int(no_count.sum())
    frac = n_miss / n_a if n_a > 0 else 0.0
    if frac > 0.05:
        raise CircInputError(
            "circCountsMissing",
            f"{n_miss} of {n_a} angle cells ({100 * frac:.1f}%) have no usable event count. Above "
            "5% that signature means the wrong file was supplied, not sparse data: check that "
            "the count files correspond to the same detection run as the angle files.")
    if n_miss > 0:
        warnings.warn(f"{n_miss} of {n_a} angle cells ({100 * frac:.2f}%) have no usable event "
                      "count; those angles are dropped because the count adjustment cannot be "
                      "applied to them.", stacklevel=2)
    A[no_count] = np.nan

    mode = str(mode).lower()
    cuse = np.where(usable_count, C, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "per_subject":
            logC = np.repeat(np.log(np.nanmean(cuse, axis=1))[:, None], n_ch, axis=1)
        elif mode == "per_channel":
            logC = np.log(cuse)
        else:
            raise CircInputError("circCountCovariateMode",
                                 "mode must be 'per_channel' or 'per_subject'.")

    pb = _point_biserial(logC, gi)
    pb_abs_max = float(np.nanmax(np.abs(pb))) if np.isfinite(pb).any() else np.nan
    pb_median = float(np.nanmedian(pb)) if np.isfinite(pb).any() else np.nan
    diagnostic = (
        "--- Event-count / group confound diagnostic (read this first) ---\n"
        f"  point-biserial r(log count, group): median {pb_median:.2f}, max |r| "
        f"{pb_abs_max:.2f}\n"
        "  |r| above ~0.5 means most of the group contrast is being absorbed by the\n"
        "  covariate and the analysis is near-uninformative whatever p emerges.\n"
        f"  channels with |r| > 0.5: {int((np.abs(pb) > 0.5).sum())}/{n_ch}")

    sd = _nanstd1(logC)
    usecov = ~(sd < 0.05) & np.isfinite(logC).any(axis=0)
    n_const = int((~usecov).sum())
    disjoint = np.zeros(n_ch, bool)
    for ch in range(n_ch):
        a = logC[gi == 1, ch]
        b = logC[gi == 0, ch]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size == 0 or b.size == 0:
            disjoint[ch] = True
            continue
        disjoint[ch] = (a.min() > b.max()) or (b.min() > a.max())
    collinear = usecov & (disjoint | (np.abs(pb) > collinearity_max))
    if collinear.mean() > 0.25:
        raise CircInputError(
            "circCountCollinear",
            f"log(event count) is collinear with group at {int(collinear.sum())} of {n_ch} "
            f"channels ({100 * collinear.mean():.0f}%, threshold 25%): |point-biserial| > "
            f"{collinearity_max:.2f} or the two groups occupy disjoint count ranges. The count "
            "adjustment is simply not estimable there, and no covariate can separate a phase "
            "effect from a detection-rate effect. The only valid route is Helfrich-style "
            "event-count subsampling: draw the same number of events per subject in both "
            "groups before computing the preferred phase, then re-run.")
    usecov = usecov & ~collinear
    report = {"mode": mode, "n_dropped_constant": n_const,
              "n_collinear": int(collinear.sum()), "pb_max": pb_abs_max,
              "pb_median": pb_median, "pb": pb, "n_cells_no_count": n_miss,
              "frac_cells_no_count": frac, "collinearity_max": collinearity_max,
              "diagnostic": diagnostic}
    return CountCovariate(A, logC, usecov, collinear, report)


# --------------------------------------------------------------------------- Freedman-Lane

def fl_context(A, logC, usecov, evaluable) -> FLContext:
    """Freedman-Lane context on the stacked response ``[cos A, sin A]`` (n, 2 * n_ch).

    Port of ``snpm_circ_fl_context.m``. The nuisance is ``[1, logC[:, ch]]`` where ``usecov``
    (the caller guarantees those columns are complete) and the intercept elsewhere, fitted once.
    A permutation is ``fit + resid[pe]`` with ONE row index for all 2 * n_ch columns
    (``permuted_embedding``); indices come from ``perm.freedman_lane_indices(n, n_perm, 'free',
    ctx.blocks, rng)``. ``ctx.evaluable`` is the (n_ch,) channel mask, not per stacked column.
    """
    A = _as_angles(A)
    n, n_ch = A.shape
    logC = np.full((n, n_ch), np.nan) if logC is None or np.size(logC) == 0 else \
        np.asarray(logC, dtype=float)
    usecov = np.asarray(usecov, dtype=bool).ravel()
    evaluable = np.asarray(evaluable, dtype=bool).ravel()
    Y = np.hstack([np.cos(A), np.sin(A)])
    fit = np.full((n, 2 * n_ch), np.nan)
    resid = np.full((n, 2 * n_ch), np.nan)

    plain = np.flatnonzero(evaluable & ~usecov)
    if plain.size:
        idx = np.concatenate([plain, plain + n_ch])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu = np.nanmean(Y[:, idx], axis=0)
        fit[:, idx] = mu
        resid[:, idx] = Y[:, idx] - mu
    cch = np.flatnonzero(evaluable & usecov)
    if cch.size:
        idx = np.concatenate([cch, cch + n_ch])
        c = np.hstack([logC[:, cch], logC[:, cch]])
        y = Y[:, idx]
        cb, yb = c.mean(axis=0), y.mean(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            b1 = ((c - cb) * (y - yb)).sum(axis=0) / ((c - cb) ** 2).sum(axis=0)
        b0 = yb - b1 * cb
        fit[:, idx] = b0 + b1 * c
        resid[:, idx] = y - fit[:, idx]
    fit[np.isnan(fit)] = 0.0
    return FLContext(fit, resid, np.ones(n), "free", evaluable)


def permuted_embedding(ctx: FLContext, pe) -> tuple[np.ndarray, np.ndarray]:
    """One Freedman-Lane realisation split back into (cos, sin), each (n, n_ch)."""
    Y = ctx.fit + ctx.resid[np.asarray(pe, dtype=int)]
    n_ch = Y.shape[1] // 2
    return Y[:, :n_ch], Y[:, n_ch:]


# --------------------------------------------------------------------------- Tier-1 linearise

_MEASURES = {"signed": "circ_signed_deviation", "signed_deviation": "circ_signed_deviation",
             "circ_signed_deviation": "circ_signed_deviation",
             "unsigned": "circ_unsigned_distance", "unsigned_distance": "circ_unsigned_distance",
             "circ_unsigned_distance": "circ_unsigned_distance"}


def _sanitize(v) -> str:
    if v is None or (isinstance(v, str) and v == ""):
        return "none"
    return re.sub(r"[^A-Za-z0-9_.:+\-]", "_", str(v))


def _g(v: float) -> str:
    """MATLAB %g."""
    return f"{v:g}"


def encode_stamp(stamp: dict) -> str:
    return "|".join(f"{k}={v}" for k, v in stamp.items())


def decode_stamp(s: str) -> dict:
    return dict(kv.split("=", 1) for kv in s.split("|"))


def linearise(A, measure: str, *, theta0: float = 0.0, max_dev_deg: float = 150.0,
              min_R: float = 0.3, units: str = "rad", output_units: str = "rad",
              convention: str = "radians_ccw_from_zero", count_file: str = "") \
        -> tuple[np.ndarray, dict]:
    """Tier-1 circular-to-linear transform, port of ``snpm_circ_linearise.m``.

    ``signed``: deviation from the pooled grand mean direction per channel, in (-pi, pi].
    ``unsigned``: |deviation| from the a-priori ``theta0``; needs an event-count covariate
    downstream (stamped ``requires_count_covariate=1``). A channel failing any guard (any missing
    subject, a subject more than ``max_dev_deg`` from the pooled mean, pooled R < ``min_R``) is
    returned as a whole NaN column, never per-cell NaN.
    """
    try:
        measure = _MEASURES[str(measure).lower()]
    except KeyError:
        raise CircInputError("measure", "measure must be 'signed' (circ_signed_deviation) or "
                             f"'unsigned' (circ_unsigned_distance); got '{measure}'.") from None
    units, output_units = str(units).lower(), str(output_units).lower()
    if units not in ("rad", "deg") or output_units not in ("rad", "deg"):
        raise CircInputError("units", "units and output_units must be 'rad' or 'deg'.")
    A = _as_angles(A)
    if units == "deg":
        A = A * np.pi / 180
    n_subj, n_ch = A.shape

    present = ~np.isnan(A)
    z = np.where(present, np.exp(1j * np.where(present, A, 0.0)), 0)
    npres = present.sum(axis=0)
    rsum = z.sum(axis=0)
    theta_ref = np.angle(rsum)
    R = np.abs(rsum) / np.maximum(npres, 1)
    R[npres == 0] = np.nan
    theta_ref[npres == 0] = np.nan
    dev_pooled = np.angle(np.exp(1j * (A - theta_ref)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        max_abs_dev = np.nanmax(np.where(present, np.abs(dev_pooled), np.nan), axis=0) \
            if n_subj else np.full(n_ch, np.nan)
    max_abs_dev[npres == 0] = np.nan

    any_missing = (~present).any(axis=0)
    with np.errstate(invalid="ignore"):
        wrap_fail = max_abs_dev > max_dev_deg * np.pi / 180
        conc_fail = ~(R >= min_R)
    valid = ~(any_missing | wrap_fail | conc_fail)
    reason = []
    for ch in range(n_ch):
        r = []
        if any_missing[ch]:
            r.append(f"missing_subjects={n_subj - npres[ch]}")
        if wrap_fail[ch]:
            r.append(f"wraparound_max_dev={max_abs_dev[ch] * 180 / np.pi:.1f}deg")
        if conc_fail[ch]:
            r.append(f"low_concentration_R={R[ch]:.3f}")
        reason.append(";".join(r))

    if measure == "circ_signed_deviation":
        Y = dev_pooled.copy()
        rotation, ref_kind, needs = "pooled_grand_mean_per_channel", "pooled_grand_mean", 0
    else:
        Y = np.abs(np.angle(np.exp(1j * (A - theta0))))
        rotation, ref_kind, needs = f"theta0_{_g(theta0)}", "apriori", 1
    Y[:, ~valid] = np.nan
    Y[~present] = np.nan
    if output_units == "deg":
        Y = Y * 180 / np.pi

    stamp = {"circ_stamp_version": "1", "measure": measure, "units": output_units,
             "convention": _sanitize(convention), "rotation": _sanitize(rotation),
             "theta_ref": ref_kind, "requires_count_covariate": str(needs),
             "count_file": _sanitize(count_file)}
    info = {"measure": measure, "theta_ref": theta_ref, "R": R,
            "max_abs_dev_deg": max_abs_dev * 180 / np.pi, "valid": valid, "reason": reason,
            "n_dropped": int((~valid).sum()), "n_present": npres, "stamp": stamp,
            "stamp_string": encode_stamp(stamp)}
    return Y, info


def linearise_table(Y, info: dict, channel_labels=None, subject_ids=None) -> pd.DataFrame:
    """The transform as a table: optional ``Subject``, one column per channel, and the reserved
    ``circ_provenance`` column carrying the stamp on every row."""
    Y = np.asarray(Y, dtype=float)
    n_subj, n_ch = Y.shape
    labels = [f"Ch{k + 1:03d}" for k in range(n_ch)] if channel_labels is None \
        else list(channel_labels)
    if len(labels) != n_ch:
        raise CircInputError("labelCount", f"channel_labels has {len(labels)} entries, "
                             f"expected {n_ch}.")
    T = pd.DataFrame(Y, columns=labels)
    if subject_ids is not None:
        subj = list(subject_ids)
        if len(subj) != n_subj:
            raise CircInputError("subjectCount", f"subject_ids has {len(subj)} entries, "
                                 f"expected {n_subj}.")
        subj = [f"sub{int(s):03d}" if isinstance(s, (int, np.integer, float, np.floating))
                else str(s) for s in subj]
        T.insert(0, "Subject", subj)
    T["circ_provenance"] = info["stamp_string"]
    return T


# --------------------------------------------------------------------------- descriptive

def mean_direction_deg(A) -> np.ndarray:
    """Per-channel mean direction in degrees on [0, 360), NaN cells ignored
    (``circ_mean_deg`` in ``core_snpm_circ.m``)."""
    A = _as_angles(A)
    z = np.where(np.isnan(A), 0, np.exp(1j * np.nan_to_num(A)))
    return np.mod(np.degrees(np.angle(z.sum(axis=0))), 360.0)


def _summary(a: np.ndarray, label: str) -> dict:
    n = a.size
    if n == 0:
        return {"label": label, "n": 0, "mean_deg": np.nan, "sd_deg": np.nan, "R": np.nan,
                    "rayleigh_z": np.nan, "rayleigh_p": np.nan}
    s = np.exp(1j * a).sum()
    r = abs(s) / n
    mu = float(np.mod(np.degrees(np.angle(s)), 360.0))
    sd = float(np.degrees(np.sqrt(-2 * np.log(max(r, _REALMIN)))))   # circular SD
    z = n * r ** 2
    pv = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * r) ** 2)) - (1 + 2 * n))
    return {"label": label, "n": int(n), "mean_deg": mu, "sd_deg": sd, "R": float(r),
            "rayleigh_z": float(z), "rayleigh_p": float(min(max(pv, 0.0), 1.0))}


def descriptive(A, gi=None, evaluable=None) -> dict:
    """No-inference panel: per group, every finite angle over the evaluable channels pooled into
    mean direction, circular SD ``sqrt(-2 ln R)`` (degrees), R, and Rayleigh z and p (Zar's
    closed form). ``gi`` None means one group, 'All subjects'."""
    A = _as_angles(A)
    ev = np.ones(A.shape[1], bool) if evaluable is None else np.asarray(evaluable, bool).ravel()
    if gi is None:
        groups = [(np.ones(A.shape[0], bool), "All subjects")]
    else:
        gi = np.asarray(gi, dtype=float).ravel()
        groups = [(gi == 1, "Group A"), (gi == 0, "Group B")]
    out = []
    for rows, label in groups:
        a = A[np.ix_(rows, ev)].ravel()
        out.append(_summary(a[np.isfinite(a)], label))
    return {"group": out, "n_channels": int(ev.sum()), "note": DESCRIPTIVE_NOTE}
