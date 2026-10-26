"""The ``[turtlewave]`` section of a run config: databases to the analysis files, in one run.

``prepare`` reads the TurtleWave databases named in ``params['turtlewave']``, builds the table
of one measure with ``turtlewave.event_table``, splits or joins it as the analysis needs, writes
every table it builds to the run's own tables folder (``<run>_tables/`` beside the results,
kept, with ``recordings.csv``), and returns
the parameters with ``data1_file``/``data2_file`` or ``data_file``/``meta_cols`` filled, so the
pipeline runs exactly as it would on hand-written files.
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path

import pandas as pd

from . import tables
from . import turtlewave as tw

KEYS = ("folder", "event_type", "method", "stage", "freq_lower", "freq_upper", "measure",
        "subject_from", "subjects_file")
_SCOPE = ("method", "stage", "freq_lower", "freq_upper")
_ONE_ROW = ("onesampleT", "unpairedT", "correlationP", "correlationS")
_JOINED = ("anova1", "ancova", "regression", "rmanova", "mixed2way")
_SUPPORTED = ("pairedT", *_ONE_ROW, *_JOINED)
ROUTING = ("sessions", "groups", "behavior_col")     # top-level keys read only with the section


def _key(subject) -> str:
    """The subject ID used to match rows: the text without a leading ``sub-``/``sub_`` and in
    lower case, the rule ``find_recordings`` uses. IDs are compared as text, so ``01`` and ``1``
    are different subjects."""
    return re.sub(r"^sub[-_]?", "", str(subject).strip().lower())


def _keys(values, what) -> list[str]:
    ids = [str(v) for v in values]
    keys = [_key(s) for s in ids]
    seen: dict[str, str] = {}
    for s, k in zip(ids, keys):
        if k in seen and seen[k] != s:
            raise ValueError(f"{what}: subjects '{seen[k]}' and '{s}' name the same subject "
                             "once a leading 'sub-' and letter case are ignored.")
        seen.setdefault(k, s)
    return keys


def _safe(name) -> str:
    return re.sub(r"[^\w.-]+", "_", str(name))


def _channels_only(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns="session")


def read_subjects(path, as_text: bool = False) -> tuple[pd.DataFrame, str]:
    """A subjects file and the name of its subject column. The subject column is read as text,
    so 01 stays 01 and matches a database named 01; with ``as_text`` every column is (an empty
    cell is missing), which is how group labels are compared. The window reads the file with
    this function too, so the groups it offers are the ones compared here."""
    path = Path(path)
    excel = path.suffix.lower() in (".xlsx", ".xls", ".xlsm")
    head = pd.read_excel(path, nrows=0) if excel else pd.read_csv(path, nrows=0)
    col = tables.subject_column(head.columns)
    if col is None:
        raise ValueError(f"{path} has no subject column (a column whose name contains "
                         "'subject').")
    if not as_text:
        info = (pd.read_excel(path, dtype={col: str}) if excel
                else tables.read_csv(path, dtype={col: str}))
        return info, col
    # Only an empty (or blank) cell is missing: a group called NA or None is a label.
    info = (pd.read_excel(path, dtype=str, keep_default_na=False) if excel
            else tables.read_csv(path, dtype=str, keep_default_na=False))
    return info.apply(lambda c: c.where(c.str.strip() != "")), col


def _subjects(path, table, keys):
    """The subjects file indexed by subject key, refusing any subject on one side only; and the
    same file read as text, for comparing group labels."""
    info, col = read_subjects(path)
    text = read_subjects(path, as_text=True)[0]
    skeys = _keys(info[col], str(path))
    if len(set(skeys)) < len(skeys):
        dup = sorted({k for k in skeys if skeys.count(k) > 1})
        raise ValueError(f"{path} lists subject(s) more than once: {', '.join(dup)}.")
    no_info = sorted({s for s, k in zip(table["Subject"], keys) if k not in skeys})
    no_db = sorted({str(s) for s, k in zip(info[col], skeys) if k not in set(keys)})
    if no_info or no_db:
        parts = ([f"databases with no row in {Path(path).name}: {', '.join(no_info)}"]
                 if no_info else []) + \
                ([f"rows of {Path(path).name} with no database: {', '.join(no_db)}"]
                 if no_db else [])
        raise ValueError("The databases and the subjects file must name the same subjects; "
                         + "; ".join(parts) + ". Add or remove them so both agree.")
    info = info.assign(_key=skeys).set_index("_key").drop(columns=col)
    clash = [c for c in info.columns if c in table.columns]
    if clash:
        raise ValueError(f"{path} has column(s) that the event table already has: "
                         f"{', '.join(map(str, clash))}.")
    return info, text.assign(_key=skeys).set_index("_key").drop(columns=col)


def _one_row_per_subject(table, keys, comparison):
    if len(set(keys)) < len(keys):
        dup = sorted({s for s, k in zip(table["Subject"], keys) if keys.count(k) > 1})
        raise ValueError(
            f"{comparison} needs one recording per subject, but these subjects have more than "
            f"one session: {', '.join(map(str, dup))}. Name the session to analyse with "
            "sessions = [\"...\"], or use pairedT, rmanova or mixed2way to compare sessions.")


def _run_scope(recordings, event_type, scope) -> tuple[list[str], list[str]]:
    """The methods and bands of the detection runs the request selected in these databases."""
    methods, bands = set(), set()
    for r in recordings:
        with sqlite3.connect(f"file:{r.path}?mode=ro", uri=True) as con:
            _, _, rows = tw._run_scope(con, r, event_type, scope.get("method"),
                                       scope.get("stage"), scope.get("freq_lower"),
                                       scope.get("freq_upper"))
        for m, _, lo, hi in rows:
            methods.add(str(m))
            bands.add(f"{lo:g}-{hi:g} Hz")
    return sorted(methods), sorted(bands)


def _density_version() -> str:
    from importlib import metadata
    try:
        return metadata.version("turtlewave-hdeeg")
    except metadata.PackageNotFoundError:
        return ""


def prepare(params: dict, out: Path):
    """Build the analysis files from ``params['turtlewave']`` and write them to folder ``out``.

    Returns ``(params, source, notes)``: the parameters with the data files filled, the
    provenance written into the summary and the report, and the notes (subjects left out) shown
    with the results.
    """
    params = dict(params)
    cfg = dict(params.pop("turtlewave"))
    routing = {k: params.pop(k) for k in ROUTING if k in params}
    comparison = params["comparison"]
    given = [k for k in ("data_file", "data1_file", "data2_file") if params.get(k)]
    if given:
        raise ValueError(f"A [turtlewave] section builds the data files itself; remove "
                         f"{', '.join(given)} from the config.")
    unknown = [k for k in cfg if k not in KEYS]
    if unknown:
        raise ValueError(f"Unknown key(s) in [turtlewave]: {', '.join(unknown)}. "
                         f"Known keys: {', '.join(KEYS)}.")
    if comparison == "mixedmodel" or comparison.startswith("circ_"):
        raise ValueError(f"{comparison} cannot be run from a [turtlewave] section: it needs "
                         "trial-level rows (mixedmodel) or phase angles (circular analyses), "
                         "and the section builds one value per channel and recording. Build "
                         "those files yourself and name them in the config.")
    if comparison not in _SUPPORTED:
        raise ValueError(f"unsupported comparison {comparison!r} for a [turtlewave] section")
    for k in ("folder", "event_type", "measure"):
        if not cfg.get(k):
            raise ValueError(f"[turtlewave] needs '{k}'.")
    if params.get("meta_cols"):
        raise ValueError("With a [turtlewave] section meta_cols is filled automatically; remove "
                         "it from the config.")
    if params.get("use_covariates") and not params.get("covariate_file"):
        raise ValueError("use_covariates needs covariate_file (Subject plus the covariates); "
                         "the covariates of a correlation are not taken from subjects_file.")

    recordings = tw.find_recordings(cfg["folder"], cfg.get("subject_from", "auto"))
    scope = {k: cfg[k] for k in _SCOPE if cfg.get(k) is not None}
    measure, event_type = cfg["measure"], cfg["event_type"]
    table, manifest = tw.event_table(recordings, measure, event_type, **scope)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"eventStat_{measure}_{event_type}"
    table.to_csv(out / f"{stem}.csv", index=False)
    manifest.to_csv(out / "recordings.csv", index=False)

    notes, dropped = [], []
    sessions = routing.get("sessions")
    available = sorted(set(table["session"]))
    if sessions is not None:
        sessions = [str(s) for s in sessions]
        absent = [s for s in sessions if s not in available]
        if absent:
            raise ValueError(f"Session(s) {', '.join(absent)} are not among the databases' "
                             f"sessions ({', '.join(s or '(none)' for s in available)}).")
    if comparison == "pairedT" and (sessions is None or len(sessions) != 2):
        raise ValueError("pairedT from a [turtlewave] section needs sessions = [\"<first>\", "
                         f"\"<second>\"]; the databases hold sessions "
                         f"{', '.join(s or '(none)' for s in available)}.")
    keys = _keys(table["Subject"], "The databases")
    # The analysis reads the subject column back from CSV, where 01 and 1 (or 7 and 007) both
    # become the number 1 (7); two such subjects would be merged there, so refuse them here.
    as_number = pd.to_numeric(pd.Series(sorted(set(table["Subject"]))), errors="coerce")
    clash = as_number[as_number.notna() & as_number.duplicated(keep=False)]
    if not clash.empty:
        names = sorted(set(table["Subject"]))
        groups = ["/".join(names[i] for i in clash.index[clash == v]) for v in clash.unique()]
        raise ValueError(f"Subject IDs that read as the same number are ambiguous: "
                         f"{', '.join(groups)}. Rename the subject folders so each subject has "
                         "one spelling.")
    info = info_text = None
    if cfg.get("subjects_file"):
        info, info_text = _subjects(cfg["subjects_file"], table, keys)
    elif comparison in ("unpairedT", "correlationP", "correlationS", "anova1", "ancova",
                        "regression", "mixed2way"):
        raise ValueError(f"{comparison} from a [turtlewave] section needs subjects_file (the "
                         "groups, covariates or behavioural scores per subject).")
    if sessions is not None and comparison != "pairedT":
        keep = table["session"].isin(sessions).to_numpy()
        table = table[keep].reset_index(drop=True)
        keys = [k for k, kept in zip(keys, keep) if kept]
    if comparison in (*_ONE_ROW, "anova1", "ancova", "regression") and \
            len(set(table["session"])) > 1:
        raise ValueError(
            f"{comparison} takes one recording per subject, but the databases come from more "
            f"than one session ({', '.join(s or '(none)' for s in sorted(set(table['session'])))})"
            ". Pooling them would mix sessions into the comparison; name the session to analyse "
            "with sessions = [\"...\"], or use pairedT, rmanova or mixed2way to compare sessions.")
    if comparison in _ONE_ROW:
        _one_row_per_subject(table, keys, comparison)
    analysed = table

    if comparison == "onesampleT":
        params["data1_file"] = str(out / f"{stem}_analysed.csv")
        _channels_only(table).to_csv(params["data1_file"], index=False)
    elif comparison == "pairedT":
        a, b = sessions
        rows = {s: {k: i for i, (k, ss) in enumerate(zip(keys, table["session"])) if ss == s}
                for s in (a, b)}
        both = [k for k in rows[a] if k in rows[b]]
        for s, other in ((a, b), (b, a)):
            for k in rows[s]:
                if k not in rows[other]:
                    dropped.append({"subject": str(table["Subject"][rows[s][k]]),
                                    "reason": f"no database for session {other}"})
        if dropped:
            msg = (f"{len(dropped)} subject(s) have only one of the sessions {a} and {b} and "
                   f"are left out: "
                   + ", ".join(f"{d['subject']} ({d['reason']})" for d in dropped) + ".")
            warnings.warn(msg, stacklevel=2)
            notes.append(msg)
        for s, role in ((a, "data1_file"), (b, "data2_file")):
            params[role] = str(out / f"{stem}_{_safe(s)}.csv")
            _channels_only(table.iloc[[rows[s][k] for k in both]]).to_csv(params[role],
                                                                          index=False)
        analysed = table.iloc[[rows[s][k] for s in (a, b) for k in both]]
    elif comparison == "unpairedT":
        group_col = params.get("group_col", "group")
        groups = routing.get("groups")
        if group_col not in info.columns:
            raise ValueError(f"group_col '{group_col}' is not a column of "
                             f"{Path(cfg['subjects_file']).name}.")
        # Group labels are compared as text, as written in the file: 01 stays 01, an empty
        # cell is no group.
        values = info_text.loc[keys, group_col].fillna("").astype(str).to_numpy()
        held = ", ".join(sorted(set(values) - {""})) or "no values"
        if groups is None or len(groups) != 2:
            raise ValueError(f"unpairedT from a [turtlewave] section needs groups = [\"<first>\", "
                             f"\"<second>\"]; '{group_col}' holds {held}.")
        groups = [str(g) for g in groups]
        if any(not g.strip() for g in groups):
            raise ValueError("groups names an empty group; name two labels of "
                             f"'{group_col}' ({held}).")
        absent = [g for g in groups if g not in set(values)]
        if absent:
            raise ValueError(f"Group(s) {', '.join(absent)} do not occur in '{group_col}' "
                             f"({held}).")
        other = [(str(s), v) for s, v in zip(table["Subject"], values) if v not in groups]
        if other:
            dropped += [{"subject": s, "reason": f"{group_col} = {v}, not compared" if v
                         else f"{group_col} is empty"} for s, v in other]
            msg = (f"{len(other)} subject(s) are in neither group {groups[0]} nor {groups[1]} "
                   f"and are left out: "
                   + ", ".join(f"{s} ({v or 'empty'})" for s, v in other) + ".")
            warnings.warn(msg, stacklevel=2)
            notes.append(msg)
        for g, role in zip(groups, ("data1_file", "data2_file")):
            params[role] = str(out / f"{stem}_{_safe(g)}.csv")
            _channels_only(table[values == g]).to_csv(params[role], index=False)
        analysed = table[pd.Series(values).isin(groups).to_numpy()]
    elif comparison in ("correlationP", "correlationS"):
        params["data1_file"] = str(out / f"{stem}_analysed.csv")
        _channels_only(table).to_csv(params["data1_file"], index=False)
        score = routing.get("behavior_col")
        if score is None:
            if info.shape[1] != 1:
                raise ValueError(f"{Path(cfg['subjects_file']).name} has several columns "
                                 f"({', '.join(map(str, info.columns))}); name the one to "
                                 "correlate with behavior_col.")
            score = info.columns[0]
        if score not in info.columns:
            raise ValueError(f"behavior_col '{score}' is not a column of "
                             f"{Path(cfg['subjects_file']).name}.")
        params["data2_file"] = str(out / f"{_safe(Path(cfg['subjects_file']).stem)}_{_safe(score)}.csv")
        pd.DataFrame({"Subject": table["Subject"],
                      score: info.loc[keys, score].to_numpy()}).to_csv(params["data2_file"],
                                                                       index=False)
    else:
        if comparison in ("rmanova", "mixed2way"):
            params.setdefault("condition_col", "session")
            params.setdefault("subject_col", "Subject")
            if params["condition_col"] == "session" and available == [""]:
                raise ValueError(f"{comparison} compares sessions, but the databases have no "
                                 "session folders (<subject>/<ses-...>/).")
        design = table
        if (design["session"] == "").all():
            design = design.drop(columns="session")
        meta = ["Subject"] + (["session"] if "session" in design.columns else [])
        parts = [design[meta]]
        if info is not None:
            parts.append(info.loc[keys].reset_index(drop=True))
            meta += [str(c) for c in info.columns]
        design = pd.concat([*parts, design.drop(columns=parts[0].columns)], axis=1)
        params["data_file"] = str(out / f"{stem}_design.csv")
        params["meta_cols"] = meta
        design.to_csv(params["data_file"], index=False)

    used = set(zip(analysed["Subject"], analysed["session"]))
    kept = [(i, r) for i, r in enumerate(recordings) if (r.subject, r.session) in used]
    manifest = manifest.iloc[[i for i, _ in kept]]
    methods, bands = _run_scope([r for _, r in kept], event_type, scope)
    interp = {f"{r.Subject} {r.session}".strip(): r.interpolated_channels.split()
              for r in manifest.itertuples() if r.interpolated_channels}
    source = {"folder": str(cfg["folder"]), "n_databases": len(kept),
              "n_databases_found": len(recordings), "event_type": event_type,
              "method": ", ".join(methods), "stage": cfg.get("stage"), "measure": measure,
              "band": ", ".join(bands), "interpolated_channels": interp,
              "dropped_subjects": dropped, "tables": str(out), **routing}
    if measure == "density":
        source["density_divided_by"] = sorted(set(manifest["stages"]))
        source["turtlewave_version"] = _density_version()
    if cfg.get("subjects_file"):
        source["subjects_file"] = str(cfg["subjects_file"])
    return params, source, notes
