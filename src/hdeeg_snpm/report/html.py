"""The HTML report: one static file, images embedded, nothing loaded from the network."""

from __future__ import annotations

from datetime import datetime

import numpy as np
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from .. import __version__, cluster_stat
from ..io.turtlewave import reject_label
from ..legacy import COMPARISONS as LEGACY_COMPARISONS
from .topoplot import data_uri, effect_figure

_METHODS = {
    "size": ("Each channel was tested, and significance was corrected across channels by "
             "permutation in two ways: threshold-free cluster enhancement (TFCE) with the maximum "
             "enhanced value over channels as the null, and a cluster test whose statistic is the "
             "number of channels in a cluster."),
    "mass": ("A general linear model was fitted at each channel. Significance was corrected "
             "across channels by permutation of residuals (Freedman-Lane), in two ways: "
             "threshold-free cluster enhancement (TFCE) with the maximum enhanced value over "
             "channels as the null, and a cluster test whose statistic is the cluster mass."),
}


def _methods(view) -> str:
    """The methods paragraph; a partial correlation adds how the covariates were controlled."""
    if view.methods:
        return view.methods
    text = _METHODS[view.cluster_statistic]
    cov = view.covariates
    if cov:
        text += (f" Covariates ({', '.join(cov['names'])}) were partialled out; covariate rank "
                 f"k = {cov['k_eff']}, error df n - 2 - k = {cov['error_df']}. Permutation "
                 f"method: {cov['method']}")
    return text


def _cluster_note(view) -> Markup:
    """The sentence under a cluster view saying how a cluster is scored. The mass wordings are
    MATLAB's (``cluster_stat``) and follow the rule ('sum' or 'mean')."""
    if view.cluster_statistic == "size":
        return Markup("The cluster statistic is the cluster <strong>size</strong> (number of "
                      "channels).")
    rule = view.cluster_rule or cluster_stat.DEFAULT
    if view.circular:
        return Markup(cluster_stat.CIRC_NOTICE[rule])
    return Markup.escape(cluster_stat.results_line(rule) + ".")


def _skipped_name(pair: str) -> str:
    """A skipped mixed2way simple effect names itself; anything else is a post-hoc pair."""
    return pair[0].upper() + pair[1:] if pair.startswith("simple effect") else f"Post-hoc {pair}"


def _p(x):
    return "n/a" if not np.isfinite(x) else f"{x:.4f}"


def _effect_context(view, setup, alpha):
    labels = setup.labels
    sig = view.significant(alpha)
    finite = np.isfinite(view.stat)
    peak = int(np.nanargmax(np.abs(np.where(finite, view.stat, np.nan)))) if finite.any() else None
    return {
        "key": view.key, "title": view.title, "stat_name": view.stat_name,
        "excluded_labels": [labels[i] for i in view.excluded],
        "excluded_groups": [(r, [labels[i] for i in idx]) for r, idx in view.excluded_by_reason()],
        "global_text": view.global_text, "global_n_channels": view.global_n_channels,
        "cluster_note": _cluster_note(view),
        "descriptive": (view.circular or {}).get("descriptive"),
        "mixed_model": view.mixed_model,
        "peak": "n/a" if peak is None else f"{abs(view.stat[peak]):.3f}",
        "peak_label": "n/a" if peak is None else labels[peak],
        "n_unc": len(sig["uncorrected"]), "n_tfce": len(sig["tfce"]), "n_cluster": len(sig["cluster"]),
        "tfce_labels": [labels[i] for i in sig["tfce"]],
        "figure": data_uri(effect_figure(view, setup, alpha)),
        "cluster_statistic": view.cluster_statistic, "cluster_threshold": view.cluster_threshold,
        "clusters": [{"number": c.number, "n": len(c.channels),
                      "statistic": f"{c.statistic:.4g}", "p": _p(c.p),
                      "significant": c.p <= alpha, "direction": c.direction, "labels": c.labels}
                     for c in view.clusters],
        "posthoc": [_effect_context(p, setup, alpha) for p in view.posthoc],
        "run_on": view.run_on, "dropped": view.dropped,
        "skipped": [(_skipped_name(pair), why) for pair, why in view.skipped],
    }


def _p_text(comparison: str) -> str:
    """How the corrected p was formed: b / N in the legacy tier, (b + 1) / (N + 1) elsewhere."""
    if comparison in LEGACY_COMPARISONS:
        return ("p-values are b / N, where N counts the observed labelling among the "
                "permutations and b is the number of them, the observed one included, whose "
                "maximum reaches the observed value.")
    return ("p-values are (b + 1) / (N + 1), where b is the number of permutations whose "
            "maximum reaches the observed value.")


def _source_rows(source) -> list[tuple[str, str]]:
    """The "Source" card of a run built from TurtleWave databases, as (label, text) rows."""
    if not source:
        return []
    stage = source.get("stage")
    rows = [("Folder", source["folder"]),
            ("Databases", f"{source['n_databases']} analysed of {source['n_databases_found']} found"),
            ("Event type", source["event_type"]),
            ("Method", source.get("method") or "not specified"),
            ("Stages", "+".join(stage) if isinstance(stage, list) else stage or "not specified"),
            ("Band", source.get("band") or "not specified"), ("Measure", source["measure"])]
    if "density_divided_by" in source:
        rows.append(("Density divided by the analysed time of",
                     ", ".join(source["density_divided_by"])))
        if "reject_types" in source:
            how = ("as detected" if source.get("reject_types_from") == "detection"
                   else "as configured")
            rows.append(("Rejected before density",
                         f"{reject_label(source['reject_types'])} ({how})"))
        rows.append(("TurtleWave version", source.get("turtlewave_version") or "unknown"))
    interp = source.get("interpolated_channels") or {}
    rows.append(("Interpolated channels set to missing",
                 "; ".join(f"{who}: {' '.join(ch)}" for who, ch in interp.items()) or "none"))
    dropped = source.get("dropped_subjects") or []
    rows.append(("Subjects left out",
                 "; ".join(f"{d['subject']} ({d['reason']})" for d in dropped) or "none"))
    if source.get("subjects_file"):
        rows.append(("Subjects file", source["subjects_file"]))
    rows.append(("Tables written to", f"{source['tables']} (with recordings.csv)"))
    return rows


def render(views, setup, params, alpha, seed, notes=(), title=None, source=None) -> str:
    env = Environment(loader=PackageLoader("hdeeg_snpm.report", "templates"),
                      autoescape=select_autoescape(["html", "j2"]))
    n_perm = sorted({v.n_permutations for v in views})
    shown = ("comparison", "tail", "datatype", "data1_file", "data2_file", "data_file",
             "covariate_file", "count1_file", "count2_file", "circ_units", "circ_convention",
             "circ_zero_offset_deg", "measure_col", "group_col", "condition_col", "subject_col",
             "predictor_col", "covariate_cols", "lmm_dv", "lmm_fixed", "lmm_random",
             "lmm_effect", "lmm_effect_type", "lmm_perm", "lmm_subject", "lmm_group")
    settings = [(k, params[k]) for k in shown if params.get(k) not in (None, "", [])]
    if views[0].cluster_rule:         # the rule in force, shown even when left at the default
        settings.append(("cluster_statistic", cluster_stat.card_value(
            views[0].cluster_rule,
            cluster_stat.CIRC_WHAT if views[0].circular else cluster_stat.GLM_WHAT)))
    return env.get_template("report.html.j2").render(
        title=title or f"hdEEG-SnPM report: {params['comparison']}",
        created=datetime.now().strftime("%Y-%m-%d %H:%M"), montage=setup.system_key,
        n_channels=len(setup.labels), methods=_methods(views[0]),
        settings=settings,
        source=_source_rows(source),
        tail_note=("Left tail: the statistic shown is Data 2 minus Data 1, so the tested "
                   "direction (Data 1 below Data 2) appears as positive values."
                   if params.get("tail") == "left" else ""),
        mixed=params["comparison"].lower() == "mixed2way",
        # skipped pairs and left-out subjects are stated under their own effect
        # the cluster rule is stated under each cluster view instead
        notes=[n for n in notes if not n.startswith(("Post-hoc ", "Simple effect ",
                                                     "Cluster statistic: "))], alpha=f"{alpha:g}", seed=seed, version=__version__,
        permutations_text=f"{' / '.join(str(n) for n in n_perm)} permutations per test.",
        p_text=_p_text(params["comparison"]),
        tfce_dh=f"{views[0].tfce_dh:g}",
        effects=[_effect_context(v, setup, alpha) for v in views])
