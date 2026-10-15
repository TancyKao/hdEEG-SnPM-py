"""Run an analysis and write its files: results (.npz + .json), Excel workbook, HTML report."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from . import __version__, pipeline
from .report import excel, html
from .views import effect_views


def _flatten(views):
    for v in views:
        yield v
        yield from v.posthoc


def results_arrays(views) -> dict[str, np.ndarray]:
    """Per-effect arrays under ``<effect key>/<name>``; channel indices are 0-based."""
    out = {}
    for v in _flatten(views):
        k = v.key
        out[f"{k}/stat"], out[f"{k}/p_uncorrected"] = v.stat, v.p_uncorrected
        out[f"{k}/tfce"], out[f"{k}/p_tfce"] = v.tfce, v.p_tfce
        out[f"{k}/excluded"] = np.asarray(v.excluded, dtype=int)
        if v.p_max is not None:
            out[f"{k}/p_max"] = v.p_max
        if v.n_analysed is not None:
            out[f"{k}/n_analysed"], out[f"{k}/n_available"] = v.n_analysed, v.n_available
        for c in v.clusters:
            out[f"{k}/cluster{c.number}"] = np.asarray(c.channels, dtype=int)
    return out


def results_summary(views, setup, params, alpha, seed, notes) -> dict:
    def one(v):
        sig = v.significant(alpha)
        return {
            "key": v.key, "title": v.title, "statistic": v.stat_name,
            "n_permutations": v.n_permutations,
            "global": {"statistic": v.global_stat, "p": v.global_p,
                       "n_channels": v.global_n_channels, "reason": v.global_reason},
            "excluded_channels": [setup.labels[i] for i in v.excluded],
            "excluded_reasons": list(v.excluded_reasons),
            "significant_channels": {name: [setup.labels[i] for i in idx]
                                     for name, idx in sig.items()},
            "cluster_statistic": v.cluster_statistic, "cluster_threshold": v.cluster_threshold,
            "clusters": [{"number": c.number, "channels": c.labels, "statistic": c.statistic,
                          "p": c.p, "direction": c.direction} for c in v.clusters],
            "posthoc": [one(p) for p in v.posthoc],
            "run_on": v.run_on, "dropped_subjects": v.dropped,
            "skipped_pairs": [{"pair": k, "reason": why} for k, why in v.skipped],
        }

    return {"package": "hdeeg-snpm", "version": __version__,
            "created": datetime.now().isoformat(timespec="seconds"),
            "params": {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                       for k, v in params.items()},
            "seed": seed, "alpha": alpha, "tfce": {"E": pipeline.TFCE_E, "H": pipeline.TFCE_H,
                                                   "dh": 0.1},
            "montage": setup.system_key, "channels": list(setup.labels),
            "notes": list(notes), "effects": [one(v) for v in views]}


def run_and_write(params: dict) -> dict[str, Path]:
    """Run ``params`` and write ``<base>_<timestamp>.npz/.json/.xlsx`` and ``_report.html`` to
    ``params['output_path']``. Returns the paths. The seed is drawn here when none is given and
    is recorded in the JSON file and the report, so any run can be repeated."""
    params = dict(params)
    seed = params.get("seed")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    params["seed"] = seed
    results = pipeline.run(params, np.random.default_rng(seed))
    setup = results.setup if hasattr(results, "setup") else results.prepared.setup
    notes = results.notes if hasattr(results, "notes") else results.prepared.notes
    views = effect_views(results)

    out_dir = Path(params.get("output_path", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(params.get("data_file") or params["data1_file"]).stem
    base = re.sub(r"\W+", "_", f"{source}_{params['comparison']}")
    base = out_dir / f"{base}_{datetime.now():%Y%m%d_%H%M%S}"
    paths = {"npz": base.with_suffix(".npz"), "json": base.with_suffix(".json"),
             "xlsx": base.with_suffix(".xlsx"), "html": Path(f"{base}_report.html")}
    np.savez_compressed(paths["npz"], **results_arrays(views))
    paths["json"].write_text(json.dumps(
        results_summary(views, setup, params, results.alpha, seed, notes), indent=2, default=float))
    excel.write_workbook(paths["xlsx"], views, setup.labels, results.alpha)
    paths["html"].write_text(html.render(views, setup, params, results.alpha, seed, notes),
                             encoding="utf-8")
    return paths
