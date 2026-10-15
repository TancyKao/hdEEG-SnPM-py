"""Threshold-free cluster enhancement over the channel graph. Ports ``ClusterEnhancement.m``.

For each channel the enhanced value is the sum, over thresholds ``h`` below the channel's
statistic, of ``cluster_size**E * h**H * dh``, where the cluster is the connected set of channels
above ``h`` that contains the channel. Positive and negative sides are enhanced separately and
the result is ``positive - negative`` (Smith & Nichols 2009).

``tfce_reference`` is a direct transcription of the MATLAB loop. ``tfce`` computes the same
numbers by adding channels from the highest threshold down and merging clusters as they join,
which avoids a connected-components search at every threshold. The two are tested against each
other and against the stored MATLAB output.

Both build the full ladder of thresholds ``dh, 2 dh, ..., max + dh`` (MATLAB's colon), whose
length grows with the largest statistic. Above ``DENSE_MAX_LEVELS`` levels :func:`tfce_event`
takes over: same integral on levels ``i * dh``, but clusters are found once per distinct number
of levels below a channel's value, and the sum of ``h ** H * dh`` over the levels in between is
taken in closed form, so time and memory no longer depend on the size of the statistic. A t with
one error degree of freedom is Cauchy-distributed, so a value of 1e7 is a legitimate draw.
"""

from __future__ import annotations

import numpy as np

from .clusters import ChannelGraph, find_clusters


def matlab_colon(a: float, d: float, b: float) -> np.ndarray:
    """``a:d:b`` for doubles with ``d > 0``, built the way MATLAB builds it.

    MATLAB fills the vector from both ends towards the middle and snaps the last element to
    ``b`` when it lies within rounding error, so the values differ in the last bit from
    ``a + d * arange(n)``.
    """
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(d)) or d <= 0 or a > b:
        return np.empty(0)
    tol = 2.0 * np.finfo(float).eps * max(abs(a), abs(b))
    n = int(np.floor((b - a) / d + 0.5))  # MATLAB round: halves away from zero
    if a + n * d - b > tol:
        n -= 1
    c = a + n * d
    if c - b > -tol:
        c = b
    out = np.empty(n + 1)
    k = np.arange(n // 2 + 1)
    out[k] = a + k * d
    out[n - k] = c - k * d
    if n % 2 == 0:
        out[n // 2] = (a + c) / 2
    return out


# Ladders up to this many levels use the dense code, unchanged, so every stored MATLAB result
# stays bit-identical; MATLAB's ClusterEnhancement.m switches at the same count.
DENSE_MAX_LEVELS = 10_000


def _top(values: np.ndarray) -> float:
    top = np.nanmax(np.abs(values))
    if np.isinf(top):
        raise ValueError("TFCE input contains an infinite statistic; exclude degenerate channels first")
    return float(top)


def _n_levels(dh: float, top: float) -> int:
    """Length of MATLAB's ``dh:dh:top+dh`` without building it (the count of :func:`matlab_colon`)."""
    a, b = dh, top + dh
    tol = 2.0 * np.finfo(float).eps * max(abs(a), abs(b))
    n = np.floor((b - a) / dh + 0.5)
    if a + n * dh - b > tol:
        n -= 1
    return int(n) + 1


def _power_run(lo: float, hi: float, H: int) -> float:
    """``sum_{i=lo}^{hi} i ** H`` for integer ``0 <= H <= 3`` as ``sum_k (lo + k) ** H``, expanded
    binomially over Faulhaber sums ``S_j(m) = sum_{k=0}^{m} k ** j``. Every term is non-negative,
    so a short run far up the ladder keeps full precision (``S(hi) - S(lo - 1)`` would not)."""
    m = hi - lo
    faulhaber = (m + 1, m * (m + 1) / 2, m * (m + 1) * (2 * m + 1) / 6, (m * (m + 1) / 2) ** 2)
    binom = ((1,), (1, 1), (1, 2, 1), (1, 3, 3, 1))[H]
    return sum(binom[j] * lo ** (H - j) * faulhaber[j] for j in range(H + 1))


def _enhance_side_event(side, graph, E, H, dh):
    n_active = np.zeros(side.size)
    ok = side > 0                       # False for NaN
    v = side[ok]
    # number of levels i * dh (i >= 1) strictly below the value; float counts so that no value
    # can overflow an integer, corrected by one either way for rounding of v / dh
    g = np.maximum(np.ceil(v / dh) - 1, 0)
    for _ in range(2):
        up = (g + 1) * dh < v
        g[up] += 1
        down = (g > 0) & ~(g * dh < v)
        g[down] -= 1
    n_active[ok] = g
    out = np.zeros(side.size)
    counts = np.unique(n_active[n_active > 0])[::-1]          # distinct counts, descending
    for j, hi in enumerate(counts):
        lo = counts[j + 1] + 1 if j + 1 < counts.size else 1.0
        # channels with n_active >= hi are active at every level lo..hi, and no other is
        weight = dh ** (H + 1) * _power_run(lo, hi, H)
        for cluster in find_clusters((n_active >= hi).astype(float), 0.5, graph):
            out[cluster] += cluster.size ** E * weight
    return out


def tfce_event(values, graph: ChannelGraph, E: float = 0.5, H: float = 2.0, dh: float = 0.1):
    """TFCE on levels ``i * dh`` whose cost does not depend on the largest statistic.

    Used automatically above ``DENSE_MAX_LEVELS``; callable directly for checks. ``H`` must be
    a whole number from 0 to 3.
    """
    if H != int(H) or not 0 <= H <= 3:
        raise ValueError(f"TFCE above {DENSE_MAX_LEVELS} levels needs a whole-number H from 0 "
                         f"to 3, got {H}")
    values = np.asarray(values, dtype=float).ravel()
    if values.size != graph.n:
        raise ValueError(f"map has {values.size} channels but the graph has {graph.n}")
    nan = np.isnan(values)
    if nan.all():
        return np.full(values.size, np.nan)
    _top(values)
    H = int(H)
    out = _enhance_side_event(np.where(values < 0, 0.0, values), graph, E, H, dh)
    if not np.all((values >= 0) | nan):
        out = out - _enhance_side_event(np.where(-values < 0, 0.0, -values), graph, E, H, dh)
    out[nan] = np.nan
    return out


def _enhance_side_reference(side, levels, graph, E, H, dh):
    contrib = np.full((levels.size, side.size), np.nan)
    for i, h in enumerate(levels):
        for cluster in find_clusters(side, h, graph):
            contrib[i, cluster] = cluster.size ** E * h ** H * dh
    return np.nansum(contrib, axis=0)


def _enhance_side_fast(side, levels, graph, E, H, dh):
    n = side.size
    # n_active[c] = number of thresholds strictly below the channel's value
    n_active = np.searchsorted(levels, np.where(np.isnan(side), -np.inf, side), side="left")
    order = np.argsort(-n_active, kind="stable")
    label = np.full(n, -1)
    size = np.zeros(n)
    contrib = np.zeros((levels.size, n))
    nbrs = graph.neighbor_lists
    pos = 0
    for i in range(levels.size - 1, -1, -1):
        while pos < n and n_active[order[pos]] > i:
            c = order[pos]
            pos += 1
            label[c] = c
            size[c] = 1
            for nb in nbrs[c]:
                a, b = label[c], label[nb]
                if b < 0 or a == b:
                    continue
                if size[a] < size[b]:
                    a, b = b, a
                label[label == b] = a
                size[a] += size[b]
        if pos:
            active = order[:pos]
            contrib[i, active] = size[label[active]] ** E * levels[i] ** H * dh
    return contrib.sum(axis=0)


def _tfce(values, graph, E, H, dh, enhance_side):
    values = np.asarray(values, dtype=float).ravel()
    if values.size != graph.n:
        raise ValueError(f"map has {values.size} channels but the graph has {graph.n}")
    nan = np.isnan(values)
    if nan.all():
        return np.full(values.size, np.nan)
    top = _top(values)
    if _n_levels(dh, top) > DENSE_MAX_LEVELS:
        return tfce_event(values, graph, E, H, dh)
    levels = matlab_colon(dh, dh, top + dh)
    pos = np.where(values < 0, 0.0, values)
    out = enhance_side(pos, levels, graph, E, H, dh)
    # A map with no negative value has an identically zero negative half.
    if not np.all((values >= 0) | nan):
        neg = np.where(-values < 0, 0.0, -values)
        out = out - enhance_side(neg, levels, graph, E, H, dh)
    out[nan] = np.nan
    return out


def tfce(values, graph: ChannelGraph, E: float = 0.5, H: float = 2.0, dh: float = 0.1):
    """TFCE-enhanced copy of a per-channel statistic map. NaN channels stay NaN."""
    return _tfce(values, graph, E, H, dh, _enhance_side_fast)


def tfce_reference(values, graph: ChannelGraph, E: float = 0.5, H: float = 2.0, dh: float = 0.1):
    """Slow transcription of ``ClusterEnhancement.m``; kept as the check for :func:`tfce`."""
    return _tfce(values, graph, E, H, dh, _enhance_side_reference)
