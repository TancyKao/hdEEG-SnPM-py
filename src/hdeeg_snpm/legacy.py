"""Permutation engines for t-tests and correlations.

Ports ``snpm_single_threshold_with_TFCE.m`` (max-statistic and TFCE correction),
``snpm_cluster_analysis.m`` (cluster-size correction) and ``snpm_corr_columns.m``.

Inputs are (subjects, channels) matrices that the caller has already transformed and masked; the
engines do not decide which channels are evaluable. Both engines take ``perms``: the permutation
index matrix to use in place of drawing one. As in MATLAB, the TFCE and cluster engines draw
separately, and a left-tailed test is run as a right-tailed test on swapped inputs, so for
``tail='left'`` the reported statistic has the sign of ``y - x``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import permutations
from math import comb, factorial

import numpy as np
from scipy import stats

from . import perm as _perm
from .clusters import ChannelGraph, find_clusters
from .tfce import tfce

COMPARISONS = ("pairedT", "unpairedT", "onesampleT", "correlationP", "correlationS")


# --------------------------------------------------------------------------- statistics

# An error sum of squares counts as zero when SS <= ZERO_SS_RTOL * sum(v ** 2), v being the
# values the statistic is computed from (uncentred, so all-zero data qualifies). That is a
# relative standard deviation of 1e-10: rounding leftovers sit near 1e-16, real data far above.
# At any labelling, observed or permuted, where the error term is zero the statistic is 0.
ZERO_SS_RTOL = 1e-20


def _count_mean_var(a: np.ndarray):
    """Column-wise n, mean, sample variance (variance of one value is 0), error sum of squares
    and uncentred sum of squares, all ignoring NaN."""
    ok = ~np.isnan(a)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, a, 0.0).sum(axis=0) / n
        ss = np.where(ok, (a - mean) ** 2, 0.0).sum(axis=0)
        var = ss / np.maximum(n - 1, 1)
    sq = np.where(ok, a, 0.0) ** 2
    return n, mean, var, ss, sq.sum(axis=0)


def paired_t(d: np.ndarray, return_undefined: bool = False):
    """One-sample t of each column of differences against zero; returns (t, df).

    Where the differences have no spread (every one identical) t is 0, at every labelling.
    ``return_undefined`` adds the mask of those channels as a third output.
    """
    n, mean, var, ss, sq = _count_mean_var(d)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / np.sqrt(var / n)
    undefined = (n >= 2) & (ss <= ZERO_SS_RTOL * sq)
    t = np.where(undefined, 0.0, t)
    return (t, n - 1, undefined) if return_undefined else (t, n - 1)


def unpaired_t(a: np.ndarray, b: np.ndarray, return_undefined: bool = False):
    """Pooled-variance two-sample t of each column, ``a`` minus ``b``; returns (t, df).

    Where the pooled within-group sum of squares is zero, t is 0 (see ``paired_t``).
    """
    na, ma, va, ssa, sqa = _count_mean_var(a)
    nb, mb, vb, ssb, sqb = _count_mean_var(b)
    df = na + nb - 2
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / df)
        t = (ma - mb) / (pooled * np.sqrt(1 / na + 1 / nb))
    undefined = (na >= 1) & (nb >= 1) & (df >= 1) & (ssa + ssb <= ZERO_SS_RTOL * (sqa + sqb))
    t = np.where(undefined, 0.0, t)
    return (t, df, undefined) if return_undefined else (t, df)


def corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns. A column with any non-finite value gives NaN.

    This is the complete-column rule at engine level: there is no pairwise deletion, so the rows
    analysed at a channel cannot depend on the permutation.
    """
    if x.shape != y.shape:
        raise ValueError(f"x is {x.shape} and y is {y.shape}; column-wise correlation needs equal sizes")
    n = x.shape[0]
    if n < 3:
        raise ValueError(f"column-wise correlation needs at least 3 rows; got {n}")
    with np.errstate(invalid="ignore", divide="ignore"):
        xc = x - x.sum(axis=0) / n
        yc = y - y.sum(axis=0) / n
        r = (xc * yc).sum(axis=0) / (np.sqrt((xc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum(axis=0)))
    r[~np.isfinite(r)] = np.nan
    over = np.abs(r) > 1
    r[over] = np.sign(r[over])
    return r


def tiedrank(a: np.ndarray) -> np.ndarray:
    """Column-wise midranks; NaN stays NaN and is left out of the ranking (MATLAB ``tiedrank``)."""
    return stats.rankdata(a, method="average", axis=0, nan_policy="omit")


_AS89 = (.2274, .2531, .1745, .0758, .1033, .3932, .0879, .0151, .0072, .0831, .0131, .00046)


def _as89(D, n):
    """Upper-tail probability of Spearman's sum of squared rank differences by Edgeworth series
    (Best and Roberts 1975, algorithm AS 89), as MATLAB ``corr`` evaluates it."""
    c = _AS89
    x = (2 * (D - 1) / ((n ** 3 - n) / 3) - 1) * np.sqrt(n - 1)
    y = x * x
    u = x * (c[0] + (c[1] + c[2] / n) / n + y * (-c[3] + (c[4] + c[5] / n) / n - y * (
        c[6] + c[7] / n - y * (c[8] - c[9] / n + y * (c[10] - c[11] * y) / n)) / n)) / n
    return np.clip(u / np.exp(0.5 * y) + stats.norm.sf(x), 0, 1)


@lru_cache(maxsize=4)
def _orders(n: int) -> np.ndarray:
    return np.array(list(permutations(range(n))), dtype=np.int8)


def _exact_two_sided(xr, yr) -> float:
    """Exact two-sided p for one pair of rank vectors, from all n! orderings (n < 10)."""
    D = ((xr[None, :] - yr[_orders(xr.size)]) ** 2).sum(axis=1)
    d = ((xr - yr) ** 2).sum()
    lower, upper = (D <= d + 0.25).mean(), (D >= d - 0.25).mean()
    return float(min(2 * min(lower, upper), 1.0))


def spearman_pvalue(r, xrank, yrank) -> np.ndarray:
    """Two-sided p-value of Spearman's rho per column, by the rule MATLAB ``corr`` uses.

    Fewer than 10 subjects: exact, from every ordering. Otherwise the Edgeworth series for
    columns without ties and the t approximation for columns with ties. scipy's ``spearmanr``
    uses the t approximation throughout, which differs in the third significant figure.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    xrank, yrank = np.asarray(xrank, dtype=float), np.asarray(yrank, dtype=float)
    xrank = xrank[:, None] if xrank.ndim == 1 else xrank
    yrank = yrank[:, None] if yrank.ndim == 1 else yrank
    n = xrank.shape[0]
    p = np.full(r.shape, np.nan)
    ok = np.isfinite(r)
    if n < 10:
        for j in np.flatnonzero(ok):
            p[j] = _exact_two_sided(xrank[:, j], yrank[:, j])
        return p

    def tied(a):
        s = np.sort(a, axis=0)
        return (np.diff(s, axis=0) == 0).any(axis=0)

    ties = tied(xrank) | tied(yrank)
    plain = ok & ~ties
    n3 = (n ** 3 - n) / 3
    D = n3 / 2 - np.round(r[plain] * n3 / 2)
    p[plain] = np.minimum(2 * _as89(np.maximum(D, n3 - D), n), 1)
    rt = r[ok & ties]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(rt) < 1, rt * np.sqrt((n - 2) / (1 - rt ** 2)), np.inf * np.sign(rt))
    p[ok & ties] = 2 * stats.t.sf(np.abs(t), n - 2)
    return p


def _t_pvalue(t, df, two_sided: bool):
    with np.errstate(invalid="ignore"):
        return 2 * stats.t.sf(np.abs(t), df) if two_sided else stats.t.sf(t, df)


# --------------------------------------------------------------------------- designs

class _Paired:
    """Paired and one-sample tests: sign-flip each subject's difference."""

    def __init__(self, x, y):
        if x.shape != y.shape:
            raise ValueError("paired and one-sample tests need matrices of the same size")
        self.d = x - y
        self.n_subj = x.shape[0]

    def default_n_perm(self):
        return 2 ** self.n_subj

    def draw(self, n_perm, rng):
        return _perm.signflip_patterns(self.n_subj, n_perm, rng)

    def stat(self, row):
        return paired_t(np.where(np.asarray(row, bool)[:, None], self.d, -self.d))[0]

    def observed(self, two_sided):
        # A channel whose observed differences are all identical has no defined t; it is NaN
        # here and reported, but stays in the permuted maps (where it is 0 or finite).
        t, df, undefined = paired_t(self.d, return_undefined=True)
        t[undefined] = np.nan
        return t, _t_pvalue(t, df, two_sided)


class _Unpaired:
    """Two independent groups: relabel subjects."""

    def __init__(self, x, y):
        self.data = np.concatenate([x, y], axis=0)
        self.n1 = x.shape[0]

    def default_n_perm(self):
        g = comb(self.data.shape[0], self.n1)
        return g if g <= 300000 else 50000

    def draw(self, n_perm, rng):
        return _perm.relabel_assignments(self.data.shape[0], self.n1, n_perm, rng)

    def stat(self, row):
        row = np.asarray(row, int)
        return unpaired_t(self.data[row[:self.n1]], self.data[row[self.n1:]])[0]

    def observed(self, two_sided):
        t, df, undefined = unpaired_t(self.data[:self.n1], self.data[self.n1:],
                                      return_undefined=True)
        t[undefined] = np.nan
        return t, _t_pvalue(t, df, two_sided)


class _Correlation:
    """Pearson or Spearman association: reorder the rows of x against a fixed y."""

    def __init__(self, x, y, spearman):
        if x.shape != y.shape:
            raise ValueError("correlation needs matrices of the same size")
        # Read the mask off the raw matrices: ranking maps -Inf to a finite rank.
        self.evaluable = np.all(np.isfinite(x) & np.isfinite(y), axis=0)
        self.x, self.y = (tiedrank(x), tiedrank(y)) if spearman else (x, y)
        self.spearman = spearman
        self.n_subj = x.shape[0]

    def default_n_perm(self):
        return min(factorial(self.n_subj), 50000)

    def draw(self, n_perm, rng):
        return _perm.row_permutations(self.n_subj, n_perm, rng)

    def stat(self, row):
        r = corr_columns(self.x[np.asarray(row, int)], self.y)
        r[~self.evaluable] = np.nan
        return r

    def observed(self, two_sided):
        r = corr_columns(self.x, self.y)
        r[~self.evaluable] = np.nan
        if self.spearman:
            return r, spearman_pvalue(r, self.x, self.y)
        df = self.n_subj - 2
        with np.errstate(invalid="ignore", divide="ignore"):
            t = r * np.sqrt(df / (1 - r ** 2))
        return r, _t_pvalue(t, df, True)


def _design(x, y, comparison, tail):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if comparison not in COMPARISONS:
        raise ValueError(f"unsupported comparison {comparison!r}; expected one of {COMPARISONS}")
    if tail not in ("both", "left", "right"):
        raise ValueError(f"tail must be 'both', 'left' or 'right', got {tail!r}")
    if tail != "both" and comparison not in ("pairedT", "unpairedT"):
        raise ValueError(f"{comparison} supports tail='both' only")
    if tail == "left":
        x, y = y, x
    if comparison in ("pairedT", "onesampleT"):
        return _Paired(x, y)
    if comparison == "unpairedT":
        return _Unpaired(x, y)
    return _Correlation(x, y, spearman=comparison == "correlationS")


def _resolve_perms(design, n_perm, rng, perms):
    if perms is not None:
        return np.asarray(perms)
    if n_perm is None:
        n_perm = design.default_n_perm()
    return design.draw(int(n_perm), rng if rng is not None else np.random.default_rng())


def _sort_desc(a: np.ndarray) -> np.ndarray:
    """Descending sort with NaN first, as MATLAB ``sort(...,'descend')``."""
    a = np.asarray(a, dtype=float)
    nan = np.isnan(a)
    return np.concatenate([a[nan], np.sort(a[~nan])[::-1]])


def _row_max(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=1)


TIE_RTOL = 1e-12


def _at_least(null, value):
    """``null >= value``, treating values equal to within rounding as equal.

    TFCE values and cluster masses are sums of a few terms on a coarse grid, so an observed value
    is often mathematically equal to a null value. Without the tolerance the last bit of the sum
    decides whether that null value counts, and the p-value moves by one or more counts.
    """
    value = np.asarray(value, dtype=float)
    slack = np.where(np.isfinite(value), TIE_RTOL * np.abs(value), 0.0)
    return null >= value - slack


def _fwe_p(observed: np.ndarray, null_max: np.ndarray, two_sided: bool) -> np.ndarray:
    """(count of null maxima >= observed, plus 1) / (N + 1); NaN where the observed is NaN."""
    obs = np.abs(observed) if two_sided else observed
    null = null_max[~np.isnan(null_max)]
    with np.errstate(invalid="ignore"):
        count = _at_least(null[None, :], obs[:, None]).sum(axis=1)
    p = (count + 1) / (null_max.size + 1)
    return np.where(np.isnan(observed), np.nan, p)


# --------------------------------------------------------------------------- engines

@dataclass
class MaxStatResult:
    """Output of :func:`single_threshold_tfce` (field names follow the MATLAB ``T`` and ``p``)."""

    real_T: np.ndarray
    real_TFCE: np.ndarray
    tMax: np.ndarray             # sorted descending
    tMaxTFCE: np.ndarray         # sorted descending
    chk_T: np.ndarray
    critical_T: float
    critical_T_TFCE: float
    p_real: np.ndarray           # uncorrected, parametric
    p_corrected: np.ndarray      # max-statistic family-wise corrected
    p_correctedTFCE: np.ndarray  # TFCE family-wise corrected
    n_perm: int
    perms: np.ndarray = field(repr=False)


def single_threshold_tfce(x, y, graph: ChannelGraph, comparison: str, tail: str = "both",
                          alpha: float = 0.05, n_perm: int | None = None, E: float = 0.5,
                          H: float = 2.0, rng: np.random.Generator | None = None,
                          perms=None) -> MaxStatResult:
    """Per-channel statistic with max-statistic and TFCE family-wise correction."""
    design = _design(x, y, comparison, tail)
    two_sided = tail == "both"
    perms = _resolve_perms(design, n_perm, rng, perms)
    n = perms.shape[0]
    maps = np.empty((n, graph.n))
    enhanced = np.empty((n, graph.n))
    for i, row in enumerate(perms):
        maps[i] = design.stat(row)
        enhanced[i] = tfce(maps[i], graph, E, H)
    real_T, p_real = design.observed(two_sided)
    real_TFCE = tfce(real_T, graph, E, H)
    if two_sided:
        maps, enhanced = np.abs(maps), np.abs(enhanced)
    t_max = _sort_desc(_row_max(maps))
    t_max_tfce = _sort_desc(_row_max(enhanced))
    crit = int(np.floor(alpha * n))
    return MaxStatResult(
        real_T=real_T, real_TFCE=real_TFCE, tMax=t_max, tMaxTFCE=t_max_tfce,
        chk_T=maps.mean(axis=0), critical_T=t_max[crit], critical_T_TFCE=t_max_tfce[crit],
        p_real=p_real, p_corrected=_fwe_p(real_T, t_max, two_sided),
        p_correctedTFCE=_fwe_p(real_TFCE, t_max_tfce, two_sided), n_perm=n, perms=perms)


@dataclass
class Cluster:
    channels: np.ndarray   # 0-based indices; empty when no channel passed this threshold
    threshold: float
    p: float
    permutations: int


@dataclass
class ClusterResult:
    clusters: list[Cluster]
    real_stat: np.ndarray
    max_cluster_sizes: np.ndarray   # sorted descending; (N,) two-sided, (N, 1) one-sided
    perms: np.ndarray = field(repr=False)


def cluster_analysis(x, y, threshold: float, graph: ChannelGraph, comparison: str,
                     tail: str = "both", n_perm: int | None = None,
                     rng: np.random.Generator | None = None, perms=None) -> ClusterResult:
    """Cluster-size permutation test at a fixed cluster-forming threshold.

    Two-sided tests form clusters below ``-threshold`` and above ``+threshold`` separately and
    score every cluster against the null of the larger of the two maximum sizes.
    """
    design = _design(x, y, comparison, tail)
    two_sided = tail == "both"
    thresholds = [-threshold, threshold] if two_sided else [threshold]
    perms = _resolve_perms(design, n_perm, rng, perms)
    n = perms.shape[0]
    sizes = np.zeros((n, len(thresholds)))
    for i, row in enumerate(perms):
        stat = design.stat(row)
        for j, thr in enumerate(thresholds):
            found = find_clusters(stat, thr, graph)
            if found:
                sizes[i, j] = max(c.size for c in found)
    real_stat, _ = design.observed(two_sided)
    clusters = []
    for thr in thresholds:
        found = find_clusters(real_stat, thr, graph) or [np.empty(0, dtype=int)]
        clusters += [Cluster(c, thr, 1.0, n) for c in found]
    if two_sided:
        null = np.sort(sizes.max(axis=1))[::-1]
        for c in clusters:
            c.p = ((null >= c.channels.size).sum() + 1) / (n + 1)
    else:
        null = np.sort(sizes, axis=0)[::-1]
        for c in clusters:
            c.p = ((null[:, 0] >= c.channels.size).sum() + 1) / (n + 1)
    return ClusterResult(clusters, real_stat, null, perms)


# --------------------------------------------------------------------------- whole-head test

@dataclass
class GlobalResult:
    stat: float
    pval: float
    channels_kept: np.ndarray   # (n_channels,) bool: the one channel set every unit is averaged over

    @property
    def n_channels(self) -> int:
        return int(self.channels_kept.sum())


def global_test(x, y, comparison: str, tail: str = "both", ncov: int = 0) -> GlobalResult:
    """Parametric test on the channel average (``global_stat_test.m``).

    Every unit of both matrices is averaged over the same channels: those finite in every row of
    both. Averaging each unit over its own available channels makes units measure different
    things whenever power varies across the scalp. ``ncov`` is the number of covariates already
    removed by residualisation; the correlation p-value then uses n - 2 - ncov degrees of freedom.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    _design(x[:1], y[:1], comparison, tail)    # argument checks only
    keep = np.isfinite(x).all(axis=0) & np.isfinite(y).all(axis=0)
    if not keep.any():
        warnings.warn("Whole-head test: no channel is usable in every analysed row of both "
                      "matrices, so there is no common set to average over.", stacklevel=2)
        return GlobalResult(np.nan, np.nan, keep)
    ax, ay = x[:, keep].mean(axis=1), y[:, keep].mean(axis=1)
    alt = {"both": "two-sided", "left": "less", "right": "greater"}[tail]
    if comparison in ("pairedT", "onesampleT"):
        res = stats.ttest_rel(ax, ay, alternative=alt)
        return GlobalResult(float(res.statistic), float(res.pvalue), keep)
    if comparison == "unpairedT":
        res = stats.ttest_ind(ax, ay, equal_var=True, alternative=alt)
        return GlobalResult(float(res.statistic), float(res.pvalue), keep)
    if comparison == "correlationS":
        rx, ry = tiedrank(ax), tiedrank(ay)
        r = float(corr_columns(rx[:, None], ry[:, None])[0])
        p = float(spearman_pvalue(r, rx, ry)[0])
    else:
        res = stats.pearsonr(ax, ay)
        r, p = float(res.statistic), float(res.pvalue)
    if ncov > 0:
        df = ax.size - 2 - ncov
        p = (np.nan if df <= 0 else
             float(2 * stats.t.sf(abs(r * np.sqrt(df / max(1 - r ** 2, np.finfo(float).eps))), df)))
    return GlobalResult(r, p, keep)
