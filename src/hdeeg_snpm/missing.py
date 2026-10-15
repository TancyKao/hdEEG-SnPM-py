"""Which channels can be tested, decided once before any permutation.

Two rules, both from ``core_snpm_analysis.m``. They are computed on the observed data and held
fixed, so the observed map and every permuted map cover the same channels.

* Correlation: a channel is tested only if it is complete (finite in every analysed subject of
  both matrices). Deleting incomplete pairs per channel inside a permutation loop would score the
  observed statistic against a null built on a different number of subjects.
* t-tests: a channel is tested on the subjects it has, unless it is degenerate under
  relabelling. Paired and one-sample tests need 2 usable pairs (one pair gives zero degrees of
  freedom and an infinite t). The unpaired test needs ``max(n1, n2) + 2`` usable subjects, since
  below that some relabelling leaves a group with a single subject.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .legacy import ZERO_SS_RTOL, tiedrank

T_TESTS = ("pairedT", "unpairedT", "onesampleT")
CORRELATIONS = ("correlationP", "correlationS")


@dataclass(frozen=True)
class ChannelRule:
    evaluable: np.ndarray       # (n_channels,) bool
    n_available: np.ndarray     # usable units per channel before masking
    min_units: int | None       # t-tests: the smallest usable count that is tested
    reason: str                 # why the missing-data rule excludes a channel
    why: np.ndarray | None = None   # (n_channels,) reason per channel, "" where evaluable

    @property
    def excluded(self) -> np.ndarray:
        return np.flatnonzero(~self.evaluable)

    @property
    def reasons(self) -> list[str]:
        """One reason per channel in ``excluded``."""
        why = self.why if self.why is not None else np.full(self.evaluable.size, self.reason, object)
        return [str(why[i]) for i in self.excluded]

    def exclude(self, mask: np.ndarray, reason) -> ChannelRule:
        """The rule with the evaluable channels in ``mask`` also excluded, for ``reason`` (one
        string, or one per channel)."""
        new = self.evaluable & np.asarray(mask, bool)
        why = (self.why.copy() if self.why is not None
               else np.where(self.evaluable, "", self.reason).astype(object))
        why[new] = np.broadcast_to(np.asarray(reason, dtype=object), why.shape)[new]
        return replace(self, evaluable=self.evaluable & ~new, why=why)


def channel_rule(comparison: str, x: np.ndarray, y: np.ndarray) -> ChannelRule:
    if comparison == "unpairedT":
        n_available = np.isfinite(x).sum(axis=0) + np.isfinite(y).sum(axis=0)
        min_units = max(x.shape[0], y.shape[0]) + 2
        reason = (f"fewer than {min_units} of the {x.shape[0] + y.shape[0]} subjects are usable, "
                  "so some relabelling would leave one group with a single subject")
        return _with_why(ChannelRule(n_available >= min_units, n_available, min_units, reason))
    usable = np.isfinite(x) & np.isfinite(y)
    n_available = usable.sum(axis=0)
    if comparison in CORRELATIONS:
        return _with_why(ChannelRule(usable.all(axis=0), n_available, None,
                                     "missing or non-finite data: the channel is not complete "
                                     "across every analysed subject"))
    if comparison in T_TESTS:
        return _with_why(ChannelRule(n_available >= 2, n_available, 2,
                                     "fewer than 2 usable pairs, so the t statistic has no "
                                     "degrees of freedom"))
    raise ValueError(f"no channel rule for comparison {comparison!r}")


def _with_why(rule: ChannelRule) -> ChannelRule:
    return replace(rule, why=np.where(rule.evaluable, "", rule.reason).astype(object))


# --------------------------------------------------------------------------- nothing to permute
# Reason texts are shared with MATLAB word for word; results are compared between the two.
NO_DIFFERENCE = "identical values in both conditions"
ALL_ZERO = "every value is zero"
CONSTANT_POOLED = "same value in every subject"
PERFECT_SEPARATION = "perfect separation, t undefined"
CONSTANT = "constant across subjects"
EXPLAINED = "fully explained by the covariates"


def _flat(v: np.ndarray) -> np.ndarray:
    """Columns whose centred sum of squares is zero by the tolerance, ignoring NaN."""
    ok = np.isfinite(v)
    n = ok.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(ok, v, 0.0).sum(axis=0) / n
        ss = np.where(ok, (v - mean) ** 2, 0.0).sum(axis=0)
    sq = (np.where(ok, v, 0.0) ** 2).sum(axis=0)
    return (n >= 1) & (ss <= ZERO_SS_RTOL * sq)


def t_test_no_variation(comparison: str, x: np.ndarray, y: np.ndarray):
    """Channels with nothing to permute in a t-test, on the usable values: (mask, reason).

    Paired and one-sample: every difference is zero (a constant non-zero difference is not
    excluded, because sign flips change it). Unpaired: the pooled values are constant.
    """
    if comparison == "unpairedT":
        return _flat(np.concatenate([x, y], axis=0)), CONSTANT_POOLED
    d = x - y
    zero = (np.where(np.isfinite(d), d, 0.0) ** 2).sum(axis=0) == 0
    return zero, ALL_ZERO if comparison == "onesampleT" else NO_DIFFERENCE


def correlation_no_variation(x, y, spearman: bool):
    """Channels where either side is constant on the matrices that are correlated (ranks for
    Spearman)."""
    if spearman:
        x, y = tiedrank(x), tiedrank(y)
    return _flat(x) | _flat(y)


def explained_by_covariates(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Columns whose residual sum of squares after the covariates is zero against the
    pre-residual sum of squares."""
    ss = (np.where(np.isfinite(after), after, 0.0) ** 2).sum(axis=0)
    sq = (np.where(np.isfinite(before), before, 0.0) ** 2).sum(axis=0)
    return np.isfinite(after).all(axis=0) & (ss <= ZERO_SS_RTOL * sq)


def n_analysed(comparison: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Units the statistic is computed on, per channel, after masking."""
    if comparison == "unpairedT":
        return np.isfinite(x).sum(axis=0) + np.isfinite(y).sum(axis=0)
    return (np.isfinite(x) & np.isfinite(y)).sum(axis=0)
