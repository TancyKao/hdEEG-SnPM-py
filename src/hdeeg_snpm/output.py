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
        for name, a in v.arrays.items():
            out[f"{k}/{name}"] = np.asarray(a)
    return out


def results_summary(views, setup, params, alpha, seed, notes, source=None) -> dict:
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
            **({"covariates": v.covariates} if v.covariates else {}),
            **({"circular": v.circular} if v.circular else {}),
            **({"mixed_model": v.mixed_model} if v.mixed_model else {}),
        }

    return {"package": "hdeeg-snpm", "version": __version__,
            "created": datetime.now().isoformat(timespec="seconds"),
            "params": {k: (list(v) if isinstance(v, (list, tuple, np.ndarray)) else v)
                       for k, v in params.items()},
            "seed": seed, "alpha": alpha, "tfce": {"E": pipeline.TFCE_E, "H": pipeline.TFCE_H,
                                                   "dh": views[0].tfce_dh},
            "montage": setup.system_key, "channels": list(setup.labels),
            "notes": list(notes), **({"source": source} if source else {}),
            "effects": [one(v) for v in views]}


def _base(out_dir: Path, stem: str, comparison: str, stamp: str) -> Path:
    name = re.sub(r"\W+", "_", f"{stem}_{comparison}")
    return out_dir / f"{name}_{stamp}"


def run_and_write(params: dict) -> dict[str, Path]:
    """Run ``params`` and write ``<base>_<timestamp>.npz/.json/.xlsx`` and ``_report.html`` to
    ``params['output_path']``. Returns the paths. The seed is drawn here when none is given and
    is recorded in the JSON file and the report, so any run can be repeated. A ``turtlewave``
    table first builds the data files from TurtleWave databases (``io.turtlewave_config``)."""
    params = dict(params)
    out_dir = Path(params.get("output_path", "."))
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    source, source_notes = None, []
    if "turtlewave" in params:
        from .io import turtlewave_config
        tw = params["turtlewave"]
        base = _base(out_dir, f"eventStat_{tw.get('measure')}_{tw.get('event_type')}",
                     params["comparison"], stamp)
        params, source, source_notes = turtlewave_config.prepare(params, f"{base}_tables")
    seed = params.get("seed")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    params["seed"] = seed
    results = pipeline.run(params, np.random.default_rng(seed))
    setup = results.setup if hasattr(results, "setup") else results.prepared.setup
    notes = results.notes if hasattr(results, "notes") else results.prepared.notes
    notes = [*source_notes, *notes]
    views = effect_views(results)

    out_dir.mkdir(parents=True, exist_ok=True)
    if source is None:
        base = _base(out_dir, Path(params.get("data_file") or params["data1_file"]).stem,
                     params["comparison"], stamp)
    paths = {"npz": base.with_suffix(".npz"), "json": base.with_suffix(".json"),
             "xlsx": base.with_suffix(".xlsx"), "html": Path(f"{base}_report.html")}
    np.savez_compressed(paths["npz"], **results_arrays(views))
    paths["json"].write_text(json.dumps(
        results_summary(views, setup, params, results.alpha, seed, notes, source), indent=2,
        default=float))
    excel.write_workbook(paths["xlsx"], views, setup.labels, results.alpha)
    paths["html"].write_text(html.render(views, setup, params, results.alpha, seed, notes,
                                         source=source),
                             encoding="utf-8")
    return paths
