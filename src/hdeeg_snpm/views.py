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
    statistic: float            # number of channels (size), or summed Wald (mass; divided by
                                # the number of channels when cluster_rule is 'mean')
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
    covariates: dict | None = None        # partial correlation: names, k_eff, error_df, method
    tfce_dh: float = 0.1                  # TFCE integration step
    methods: str = ""                     # methods paragraph, when the tier writes its own
    circular: dict | None = None          # circular tier: import, gate, descriptive panel, ...
    mixed_model: dict | None = None       # mixed-model tier: formula, scheme, rows dropped, ...
    arrays: dict = field(default_factory=dict)   # extra per-channel arrays for the results file
    cluster_rule: str = ""                # how a mass is scored: 'sum' or 'mean' ('' for size)

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
        n_available=prep.rule.n_available,
        covariates=prep.covariates.as_dict() if prep.covariates else None)


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
        p_tfce=r.p_correctedTFCE, cluster_statistic="mass", cluster_rule=r.cluster_statistic,
        cluster_threshold=f"uncorrected p < {alpha:g}", clusters=_number(clusters),
        n_permutations=r.n_perm, excluded=r.excluded_channels,
        excluded_reasons=list(r.excluded_reasons),
        global_text=text, global_stat=stat, global_p=p, global_reason=why,
        global_n_channels=int(loop.global_mask.sum()),
        run_on=loop.units, dropped=list(loop.dropped), skipped=list(loop.skipped))


_CIRC_METHODS = {
    "circ_phase_group": (
        "At each channel the two groups' preferred phases were compared with a two-sample "
        "Hotelling T² test on the (cos, sin) embedding of the angle, adjusted for log(event "
        "count) at that channel where the count varies. Significance was corrected across "
        "channels by permutation of residuals (Freedman-Lane on the stacked cos and sin "
        "response, one row index for both)"),
    "circ_phase_group_u2": (
        "At each channel the two groups' phase distributions were compared with Watson's "
        "two-sample U² test, which cannot adjust for event count. Significance was corrected "
        "across channels by permutation of the group labels against the fixed angles"),
    "circ_corrAngLinear": (
        "At each channel the association between the angle and the linear measure was tested "
        "with the circular-linear F of the measure on [1, cos, sin]. Significance was corrected "
        "across channels by permuting the measure (Freedman-Lane with the intercept as the only "
        "nuisance) against the fixed angles"),
}


def _plain(v):
    """JSON-ready copy: numpy scalars and arrays to Python types."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return [_plain(x) for x in v.tolist()]
    if isinstance(v, np.generic):
        return v.item()
    return v


def _circ_view(res) -> EffectView:
    from .circ import mean_direction_deg
    from .circ_pipeline import STAT_LABELS, effect_label, exclusion_reason
    from .cluster_stat import circ_recorded

    prep, r, labels = res.prepared, res.result, res.setup.labels
    rule = res.cluster_statistic
    name = "U2" if prep.comparison == "circ_phase_group_u2" else "F"
    clusters = [ClusterView(0, c.channels, [labels[i] for i in c.channels], float(c.mass), c.p)
                for c in r.clusters if c.channels.size]
    imp = prep.imp
    methods = (_CIRC_METHODS[prep.comparison] + ", in two ways: threshold-free cluster "
               "enhancement (TFCE) with the maximum enhanced value over channels as the null, and "
               f"a cluster test whose statistic is the cluster mass ({circ_recorded(rule)}; "
               "FieldTrip's maxsize, used by Helfrich et al., differs). "
               f"Angles were declared in {imp['units']} with zero convention "
               f"{imp['convention']}; a rotation of {imp['rotation_deg']:+.1f} deg was applied so "
               "that 0 is the slow-oscillation up-state peak. "
               + ("A channel is tested only with at least 8 subjects per group, a count not "
                  "collinear with group and, where the count covariate is used, no missing angle"
                  if prep.is_group else
                  "A channel is tested only with at least 8 subjects with both an angle and the "
                  "measure")
               + "; that set is fixed before the first permutation. There is no whole-head test "
               "for angles; the panel below is descriptive only.")
    inv = dict(prep.inversion, channels=[labels[i] for i in prep.inversion["channels"]])
    cov = {k: v for k, v in prep.covariate.items() if k != "pb"}
    circular = _plain({
        "comparison": prep.comparison, "statistic": prep.statistic,
        "statistic_label": STAT_LABELS[prep.comparison], "effect_label": effect_label(prep),
        "import": imp, "rotation_deg": imp["rotation_deg"], "convention": imp["convention"],
        "units": imp["units"], "covariate": cov, "gate_min_per_group": 8,
        "cluster_statistic": circ_recorded(rule), "cluster_statistic_rule": rule,
        "exclusion_reason": exclusion_reason(prep.comparison),
        "n_group1": prep.n1g, "n_group2": prep.n2g, "measure_col": prep.measure_col,
        **({} if prep.is_group else {"measure_nan_removed": prep.measure_nan_removed,
                                     "dropped_subjects": list(prep.dropped_subjects)}),
        "subjects": list(prep.subjects), "perm_type": res.engine.perm_type,
        "inversion_check": inv, "descriptive": res.descriptive,
        "whole_head_test": "none (deliberate: no valid whole-head mean of angles)"})
    # mean direction per channel (degrees): group A then group B, or all subjects
    mean_dir = (np.vstack([mean_direction_deg(prep.A[prep.gi == 1]),
                           mean_direction_deg(prep.A[prep.gi == 0])])
                if prep.is_group else mean_direction_deg(prep.A)[None, :])
    arrays = {"evaluable": prep.evaluable, "usecov": prep.usecov, "collinear": prep.collinear,
              "n_finite": prep.n_finite, "mean_direction_deg": mean_dir}
    if prep.is_group:
        arrays.update(n_group1=prep.n_group1, n_group2=prep.n_group2,
                      point_biserial=prep.covariate["pb"])
    if res.engine.rho is not None:
        arrays["rho"] = res.engine.rho
    return EffectView(
        key="main", title=f"{prep.comparison}: {STAT_LABELS[prep.comparison]}", stat_name=name,
        stat=r.real_T, p_uncorrected=r.p_real, tfce=r.real_TFCE, p_tfce=r.p_correctedTFCE,
        cluster_statistic="mass", cluster_rule=rule,
        cluster_threshold=f"uncorrected p < {res.alpha:g}",
        clusters=_number(clusters), n_permutations=r.n_perm, excluded=prep.excluded,
        excluded_reasons=list(prep.excluded_reasons), global_text="", global_n_channels=0,
        n_analysed=prep.per_channel_n, n_available=np.isfinite(prep.A_imported).sum(axis=0),
        tfce_dh=prep.tfce_dh, methods=methods, circular=circular, arrays=arrays)


def _lmm_view(res) -> EffectView:
    from .cluster_stat import card_value, rule_text
    from .lmm import dropped_text

    spec, labels, T = res.spec, res.setup.labels, res.T
    rule = res.clusters.cluster_statistic
    name = "F" if spec.effect_type == "factor" else "t"
    clusters = [ClusterView(0, c.channels, [labels[i] for i in c.channels], float(c.mass), c.p)
                for c in res.clusters.clusters if c.channels.size]
    scheme = ("the response shuffled within each subject's rows (within_subject)"
              if spec.perm == "within_subject" else
              "whole subjects relabelled between groups (group_label)")
    stat_text = ("the t of its coefficient" if name == "t" else
                 "the F of its coefficients (marginal Wald test divided by their number)")
    methods = (
        f"A linear mixed model, {spec.formula}, was fitted at each channel by maximum likelihood "
        f"with a random intercept for each level of {spec.grouping} (the only random structure "
        f"supported), reference coding and residual degrees of freedom (n - p). The effect of "
        f"{spec.effect} was mapped as {stat_text}. Significance was corrected across channels by "
        f"permutation, with {scheme}, in two ways, each with its own permutations: threshold-"
        "free cluster enhancement (TFCE) with the maximum enhanced value over channels as the "
        f"null, and a cluster test on channels with model p < {res.alpha:g} whose statistic is "
        f"the cluster mass, the {rule_text(rule)}. A channel is tested only if it is complete across every "
        "analysed row; that set is fixed before the first permutation.")
    d = res.dropped
    mixed = _plain({
        "formula": spec.formula, "dv": spec.dv, "fixed": spec.fixed, "random": spec.random,
        "random_structure": f"random intercept per {spec.grouping}",
        "effect": spec.effect, "effect_type": spec.effect_type, "perm_scheme": spec.perm,
        "subject_col": spec.subject, "group_col": spec.group, "coef_names": res.model_names,
        "n_trials": res.n_trials, "n_subjects": res.n_subjects,
        "dropped_rows": d.as_dict(), "dropped_text": dropped_text(d),
        "cluster_statistic": card_value(rule), "cluster_statistic_rule": rule,
        "n_degenerate_permutations": T.n_degenerate, "df": T.fit.df,
        "fit": "maximum likelihood, closed form in the variance ratio (exact optimum)"})
    df = f"{T.fit.df:g}" if name == "t" else f"{len(lmm_effect_cols(res))}, {T.fit.df:g}"
    gtext = (f"{name}({df}) = {res.global_stat:.4g}, p = {res.global_p:.4g}"
             if np.isfinite(res.global_stat) else f"{name} not computed: the fit failed")
    return EffectView(
        key="main", title=f"Mixed model: {spec.effect} ({spec.effect_type})", stat_name=name,
        stat=T.real_T, p_uncorrected=T.p_real, tfce=T.real_TFCE, p_tfce=T.p_correctedTFCE,
        cluster_statistic="mass", cluster_rule=rule, cluster_threshold=f"model p < {res.alpha:g}",
        clusters=_number(clusters), n_permutations=T.n_perm, excluded=res.excluded,
        excluded_reasons=list(res.excluded_reasons), global_text=gtext, global_stat=res.global_stat,
        global_p=res.global_p, global_reason="" if np.isfinite(res.global_stat) else "fit failed",
        global_n_channels=int(res.evaluable.sum()), methods=methods, mixed_model=mixed,
        arrays={"evaluable": res.evaluable, "within_ratio": res.within_ratio, "wald": T.fit.wald, "beta": T.fit.beta,
                "se": T.fit.se, "loglik": T.fit.loglik, "sd_subject": T.fit.sd_subject,
                "sd_residual": T.fit.sigma, "boundary": T.fit.boundary,
                "tfce_null": T.tMaxTFCE, "cluster_mass_null": res.clusters.maxMass,
                "dropped_rows": res.dropped.rows})


def lmm_effect_cols(res) -> list[int]:
    from .lmm import LMM
    return LMM(res.meta, res.spec).effect_cols


def effect_views(results) -> list[EffectView]:
    """Top-level effects in run order, each carrying its post-hoc contrasts."""
    if getattr(results, "tier", "") == "circ":
        return [_circ_view(results)]
    if getattr(results, "tier", "") == "lmm":
        return [_lmm_view(results)]
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
