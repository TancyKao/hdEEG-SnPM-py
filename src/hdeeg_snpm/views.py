"""One common description of a tested effect, whichever analysis produced it.

The writers (results file, Excel, maps, HTML) read these views and nothing else, so they do not
need to know the difference between the t-test tier and the GLM tier.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STAT_NAMES = {"pairedT": "t", "unpairedT": "t", "onesampleT": "t", "correlationP": "r",
              "correlationS": "rho"}


@dataclass
class ClusterView:
    number: int                 # 1, 2, ... in table order; the same number is drawn on the map
    channels: np.ndarray        # 0-based, analysed order
    labels: list[str]
    statistic: float            # number of channels (size) or summed Wald / n_channels (mass)
    p: float
    direction: str = ""         # 'positive' / 'negative' for signed two-sided tests


@dataclass
class EffectView:
    key: str                    # file-safe name: 'main', 'group', 'condition@HC', ...
    title: str
    stat_name: str              # 't', 'r', 'rho' or 'F'
    stat: np.ndarray
    p_uncorrected: np.ndarray
    tfce: np.ndarray
    p_tfce: np.ndarray
    cluster_statistic: str      # 'size' or 'mass'
    cluster_threshold: str      # how clusters were formed, in words
    clusters: list[ClusterView]
    n_permutations: int
    excluded: np.ndarray        # 0-based channel indices
    excluded_reasons: list[str] = field(default_factory=list)   # one per excluded channel
    p_max: np.ndarray | None = None       # max-statistic corrected p (t-test tier only)
    global_text: str = ""
    global_stat: float = np.nan
    global_p: float = np.nan
    global_reason: str = ""               # why the whole-head statistic is undefined, if it is
    global_n_channels: int = 0
    n_analysed: np.ndarray | None = None
    n_available: np.ndarray | None = None
    posthoc: list[EffectView] = field(default_factory=list)
    run_on: str = ""                      # a post-hoc pair: the subjects it used
    dropped: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)   # tests not run, with reason

    def excluded_by_reason(self) -> list[tuple[str, list[int]]]:
        """(reason, channel indices) in order of first appearance."""
        groups: dict[str, list[int]] = {}
        for i, r in zip(self.excluded, self.excluded_reasons):
            groups.setdefault(r, []).append(int(i))
        return list(groups.items())

    def significant(self, alpha: float) -> dict[str, np.ndarray]:
        with np.errstate(invalid="ignore"):
            return {
                "uncorrected": np.flatnonzero(self.p_uncorrected <= alpha),
                "tfce": np.flatnonzero(self.p_tfce <= alpha),
                "cluster": (np.concatenate([c.channels for c in self.clusters if c.p <= alpha])
                            if any(c.p <= alpha for c in self.clusters)
                            else np.empty(0, dtype=int)),
            }


def _number(clusters: list[ClusterView]) -> list[ClusterView]:
    """Smallest p first, then largest statistic; numbered in that order."""
    clusters = sorted(clusters, key=lambda c: (c.p, -c.statistic, c.channels.min()))
    for i, c in enumerate(clusters, 1):
        c.number = i
    return clusters


def _legacy_view(res) -> EffectView:
    prep, labels = res.prepared, res.prepared.setup.labels
    name = STAT_NAMES[prep.comparison]
    clusters = [ClusterView(0, c.channels, [labels[i] for i in c.channels], float(c.channels.size),
                            c.p, "negative" if c.threshold < 0 else "positive")
                for c in res.clusters.clusters if c.channels.size]
    thr = abs(res.clusters.clusters[0].threshold) if res.clusters.clusters else prep.threshold
    g = res.global_
    excluded, reasons = prep.excluded
    return EffectView(
        key="main", title=f"{prep.comparison} ({prep.tail} tail)", stat_name=name,
        stat=res.T.real_T, p_uncorrected=res.T.p_real, tfce=res.T.real_TFCE,
        p_tfce=res.T.p_correctedTFCE, cluster_statistic="size",
        cluster_threshold=f"|{name}| > {thr:.4g}" if prep.tail == "both" else f"{name} > {thr:.4g}",
        clusters=_number(clusters), n_permutations=res.T.n_perm, excluded=excluded,
        excluded_reasons=reasons, p_max=res.T.p_corrected,
        global_text=(f"{name} = {g.stat:.4g}, p = {g.pval:.4g}" if prep.global_interpretable
                     else "not interpretable for this data type (see the note above)"),
        global_stat=g.stat if prep.global_interpretable else np.nan,
        global_p=g.pval if prep.global_interpretable else np.nan,
        global_n_channels=g.n_channels, n_analysed=prep.n_analysed,
        n_available=prep.rule.n_available)


def _glm_view(loop, labels, alpha) -> EffectView:
    r = loop.result
    name = "t" if loop.contrast_type == "t" else "F"
    clusters = [ClusterView(0, c.channels, [labels[i] for i in c.channels], float(c.mass), c.p)
                for c in r.clusters if c.channels.size]
    stat, p, df1, df2, why = loop.global_test()
    df = f"{df2:g}" if name == "t" else f"{df1:g}, {df2:g}"
    text = (f"{name}({df}) not computed: {why}" if why
            else f"{name}({df}) = {stat:.4g}, p = {p:.4g}")
    role = loop.role.split("/posthoc:")
    return EffectView(
        key=loop.role.replace("/posthoc:", "__").replace(" ", "_"),
        title=role[1] if len(role) > 1 else f"{role[0]}: {loop.design.effect_label}",
        stat_name=name, stat=r.real_T, p_uncorrected=r.p_real, tfce=r.real_TFCE,
        p_tfce=r.p_correctedTFCE, cluster_statistic="mass",
        cluster_threshold=f"uncorrected p < {alpha:g}", clusters=_number(clusters),
        n_permutations=r.n_perm, excluded=r.excluded_channels,
        excluded_reasons=list(r.excluded_reasons),
        global_text=text, global_stat=stat, global_p=p, global_reason=why,
        global_n_channels=int(loop.global_mask.sum()),
        run_on=loop.units, dropped=list(loop.dropped), skipped=list(loop.skipped))


def effect_views(results) -> list[EffectView]:
    """Top-level effects in run order, each carrying its post-hoc contrasts."""
    if hasattr(results, "prepared"):
        return [_legacy_view(results)]
    labels, views = results.setup.labels, []
    for loop in results.loops:
        v = _glm_view(loop, labels, results.alpha)
        if "/posthoc:" in loop.role:
            views[-1].posthoc.append(v)
        else:
            views.append(v)
    return views
