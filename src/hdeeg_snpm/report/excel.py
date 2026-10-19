"""Excel workbook with the sheet names of the MATLAB toolbox (``func_genSnpmTable.m``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

SHEETS = ("statVal", "pVal_or_pCritVal", "pValTFCE", "effectiveN", "excludedChannels",
          "uncorrectsigch", "correctTFCEsigch", "correctSnPMsigch")


def _row(values, labels):
    return pd.DataFrame([np.round(np.asarray(values, dtype=float), 4)], columns=list(labels))


def _sig(index, labels):
    """Channel numbers are 1-based, as in the MATLAB workbook; the label is given beside each."""
    if len(index) == 0:
        return pd.DataFrame({"Var1": ["No significant chans"]})
    return pd.DataFrame({"ChannelIndex": np.asarray(index) + 1,
                         "ChannelLabel": [labels[i] for i in index]})


def sheets_for(view, labels, alpha) -> dict[str, pd.DataFrame]:
    sig = view.significant(alpha)
    out = {"statVal": _row(view.stat, labels), "pVal_or_pCritVal": _row(view.p_uncorrected, labels),
           "pValTFCE": _row(view.p_tfce, labels)}
    if view.n_analysed is not None:
        n = pd.DataFrame([view.n_analysed, view.n_available], columns=list(labels))
        n.insert(0, "Metric", ["n_analysed", "n_available"])
        out["effectiveN"] = n
    if len(view.excluded):
        out["excludedChannels"] = pd.DataFrame({
            "ChannelIndex": np.asarray(view.excluded) + 1,
            "ChannelLabel": [labels[i] for i in view.excluded],
            "Reason": list(view.excluded_reasons)})
    else:
        out["excludedChannels"] = pd.DataFrame({"ChannelLabel": ["None"],
                                                "Reason": ["All channels evaluable (no missing data)"]})
    mm = view.mixed_model
    if mm:
        d = mm["dropped_rows"]
        rows = [("formula", mm["formula"]), ("random_structure", mm["random_structure"]),
                ("effect", mm["effect"]), ("effect_type", mm["effect_type"]),
                ("statistic", view.stat_name), ("residual_df", mm["df"]),
                ("perm_scheme", mm["perm_scheme"]), ("n_trials", mm["n_trials"]),
                ("n_subjects", mm["n_subjects"]),
                ("cluster_statistic", mm["cluster_statistic"]),
                ("whole_head", view.global_text)]
        # MATLAB func_genSnpmTable: one summary line, then one row per model variable
        out["droppedRows"] = pd.DataFrame(
            [("summary", mm["dropped_text"])]
            + [(c, f"{k} missing") for c, k in zip(d["columns"], d["per_column"])],
            columns=["Item", "Value"])
        out["mixedModel"] = pd.DataFrame(rows, columns=["Item", "Value"]).astype(str)
    out["uncorrectsigch"] = _sig(sig["uncorrected"], labels)
    out["correctTFCEsigch"] = _sig(sig["tfce"], labels)
    out["correctSnPMsigch"] = _sig(sig["cluster"], labels)
    return out


def write_workbook(path, views, labels, alpha):
    """The first effect uses the plain sheet names. Further effects and post-hoc contrasts get
    the same sheets with a numeric prefix, listed on a ``contents`` sheet, which also gives each
    effect's whole-head test."""
    flat = []
    for v in views:
        flat.append(v)
        flat.extend(v.posthoc)
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        if len(flat) > 1 or any(v.skipped or v.global_reason for v in flat):
            rows = [{"Prefix": "" if i == 0 else f"{i + 1}_", "Effect": v.title,
                     "RunOn": v.run_on, "SubjectsLeftOut": ", ".join(v.dropped), "Status": "run",
                     "WholeHead": v.global_text}
                    for i, v in enumerate(flat)]
            rows += [{"Prefix": "", "Effect": f"{v.title}: {pair}", "RunOn": "",
                      "SubjectsLeftOut": "", "Status": f"not run: {why}", "WholeHead": ""}
                     for v in flat for pair, why in v.skipped]
            pd.DataFrame(rows).to_excel(xl, sheet_name="contents", index=False)
        for i, v in enumerate(flat):
            for name, frame in sheets_for(v, labels, alpha).items():
                frame.to_excel(xl, sheet_name=(name if i == 0 else f"{i + 1}_{name}")[:31],
                               index=False)
