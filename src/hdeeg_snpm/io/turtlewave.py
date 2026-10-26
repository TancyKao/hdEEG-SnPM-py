"""Read TurtleWave-hdEEG ``neural_events.db`` files into subjects-by-channels tables.

TurtleWave writes one database per recording. This module finds the databases, works out which
subject and session each belongs to, and builds one wide table per measure (density, amplitude,
duration or count) for one event type, method, stage and frequency band.

Amplitude, duration and count are plain SQL. Density is delegated to TurtleWave's own
``event_density`` (install with ``pip install hdeeg-snpm[turtlewave]``), so the rule for the
time an event count is divided by lives in one place.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DB_NAME = "neural_events.db"
MEASURES = ("density", "amplitude", "duration", "count")
_SESSION = re.compile(r"^(ses[-_]\w+|session[-_]?\w+)$", re.IGNORECASE)
_SUBJECT_TABLES = ("analysed_time", "detection_runs", "stage_durations", "sleep_cycles",
                   "pac_coupling")


@dataclass(frozen=True)
class Recording:
    path: Path
    subject: str
    session: str            # '' when the layout has no session folder
    subject_source: str     # 'database', 'folder' or 'database and folder'


def _same_subject(a: str, b: str) -> bool:
    """'sub-01ab', 'sub_01ab' and '01ab' name the same subject."""
    def norm(s):
        return re.sub(r"^sub[-_]?", "", str(s).strip().lower())
    return norm(a) == norm(b)


def _with_session(stored: str, folder_subject: str, session: str) -> bool:
    """A database ID that is the folder subject plus the session, ``sub-01ab_BL`` in
    ``sub-01ab/ses-BL`` (session with or without ``ses-``, case ignored)."""
    if not session:
        return False
    bare = re.sub(r"^ses(sion)?[-_]?", "", session, flags=re.IGNORECASE)
    return any(_same_subject(stored, f"{folder_subject}_{s}") for s in {session, bare} if s)


def _folder_ids(db: Path) -> tuple[str, str]:
    """Subject and session from the path: ``<subject>/<ses...>/[folder/]neural_events.db``, or
    without a session folder ``<subject>/<folder>/neural_events.db``."""
    for parent in list(db.parents)[:2]:      # the database's folder or the one above it
        if _SESSION.match(parent.name):
            return parent.parent.name, parent.name
    return db.parent.parent.name, ""


def _tables(con) -> set[str]:
    return {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(con, table) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _db_subjects(con) -> list[str]:
    found: list[str] = []
    for t in _SUBJECT_TABLES:
        if t in _tables(con) and "subject" in _columns(con, t):
            found += [r[0] for r in con.execute(f"SELECT DISTINCT subject FROM {t}") if r[0]]
    return list(dict.fromkeys(found))


def find_recordings(source, subject_from: str = "auto") -> list[Recording]:
    """Find every database under a root folder (or take an explicit list of paths).

    ``subject_from='auto'`` reads the subject ID stored in the database when there is one and
    the subject folder name otherwise; when both exist they must agree (ignoring a ``sub-``
    prefix) or the file is named in an error. A database ID that is the folder subject plus
    the session (``sub-01ab_BL`` in ``sub-01ab/ses-BL``) agrees, and the subject is then the
    folder's, so the sessions of one subject pair up. ``'database'`` or ``'folder'`` forces one source,
    for layouts where the folder above the database is not the subject.
    """
    if subject_from not in ("auto", "database", "folder"):
        raise ValueError("subject_from must be 'auto', 'database' or 'folder'")
    if isinstance(source, (str, Path)):
        root = Path(source)
        paths = [root] if root.is_file() else sorted(root.rglob(DB_NAME))
    else:
        paths = [Path(p) for p in source]
    if not paths:
        raise FileNotFoundError(f"No {DB_NAME} found under {source}")
    out = []
    for db in paths:
        if not db.is_file():
            raise FileNotFoundError(db)
        # abspath, not resolve: a folder of symlinks names its subjects by the links
        folder_subject, session = _folder_ids(Path(os.path.abspath(db)))
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            stored = _db_subjects(con)
        if len(stored) > 1:
            raise ValueError(f"{db} holds more than one subject ({', '.join(stored)}); one "
                             "database per recording is expected.")
        if subject_from == "folder" or (subject_from == "auto" and not stored):
            subject, how = folder_subject, "folder"
        elif subject_from == "database":
            if not stored:
                raise ValueError(f"{db} stores no subject ID; use subject_from='folder'.")
            subject, how = stored[0], "database"
        elif _with_session(stored[0], folder_subject, session):
            subject, how = folder_subject, "folder (database ID adds the session)"
        else:
            if not _same_subject(stored[0], folder_subject):
                raise ValueError(
                    f"{db}: the database says subject '{stored[0]}' but its folder says "
                    f"'{folder_subject}'. Fix the layout, or pass subject_from='database' or "
                    "'folder' to say which is right.")
            subject, how = stored[0], "database and folder"
        out.append(Recording(db, subject, session, how))
    if len(out) > 1 and len({r.subject for r in out}) == 1 and \
            all(r.subject_source == "folder" for r in out):
        warnings.warn(f"All {len(out)} databases were assigned to one subject "
                      f"('{out[0].subject}') from their folder names. If these are different "
                      "subjects, the folder layout is not <subject>/<ses-...>/; check "
                      "recordings.csv.", stacklevel=2)
    keys = [(r.subject, r.session) for r in out]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        listed = "; ".join(f"{s} {ses}".strip() + ": " + ", ".join(
            str(r.path) for r in out if (r.subject, r.session) == (s, ses)) for s, ses in sorted(dup))
        raise ValueError(f"More than one database for the same subject and session ({listed}).")
    return out


def available(recordings) -> pd.DataFrame:
    """What each database holds: one row per event type, method, stage and band."""
    rows = []
    for r in recordings:
        with sqlite3.connect(f"file:{r.path}?mode=ro", uri=True) as con:
            q = ("SELECT event_type, method, stage, freq_lower, freq_upper, COUNT(*), "
                 "COUNT(DISTINCT channel) FROM events GROUP BY 1, 2, 3, 4, 5")
            rows += [(r.subject, r.session, *row) for row in con.execute(q)]
    return pd.DataFrame(rows, columns=["Subject", "session", "event_type", "method", "stage",
                                       "freq_lower", "freq_upper", "n_events", "n_channels"])


def interpolated_channels(path) -> list[str]:
    """Channels TurtleWave recorded as interpolated (databases from version 4.5); else empty."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        if "detection_runs" not in _tables(con) or \
                "interpolated_channels" not in _columns(con, "detection_runs"):
            return []
        found: list[str] = []
        for (cell,) in con.execute("SELECT interpolated_channels FROM detection_runs"):
            if cell:
                found += json.loads(cell)
    return list(dict.fromkeys(found))


def reject_key(reject_types) -> str:
    """A reject set as TurtleWave keys it (``utils.reject_key``): names sorted and joined by
    commas, ``'Arousal,Artefact'``; ``''`` is the set that rejects nothing. Takes the text of
    a config (``"Artefact, Arousal"``; ``"none"``, any case, is the empty set, as in the
    MATLAB repository's bridge) or a list of names (``["none"]`` is the empty set too)."""
    if not isinstance(reject_types, str):
        reject_types = list(reject_types)
        if len(reject_types) == 1 and isinstance(reject_types[0], str):
            reject_types = reject_types[0]
    if isinstance(reject_types, str) and reject_types.strip().lower() == "none":
        return ""
    names = reject_types.split(",") if isinstance(reject_types, str) else list(reject_types)
    return ",".join(sorted({str(n).strip() for n in names if str(n).strip()}))


def reject_label(key: str) -> str:
    key = reject_key(key)
    return key.replace(",", ", ") if key else "nothing rejected"


def reject_written(key: str) -> str:
    """A reject key as written to recordings.csv and the summary: ``none`` for the empty set,
    as the bridge writes it."""
    return key or "none"


def stored_reject_sets(path) -> list[str]:
    """The reject sets whose analysed time (the density denominator) a database stores, as
    ``reject_key`` text: from the ``reject_types`` column (TurtleWave 4.4 on), else from the
    ``reject_artifacts``/``reject_arousals`` columns of TurtleWave 4.3 (1, 0 is
    ``'Artefact'``; 1, 1 ``'Arousal,Artefact'``; 0, 1 ``'Arousal'``; 0, 0 ``''``). Empty
    when the database stores no analysed time. A stored ``reject_types`` text that is not in
    TurtleWave's key spelling was written outside TurtleWave and is refused, naming it: two
    spellings of one set would otherwise be merged here and missed by TurtleWave."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as con:
        if "analysed_time" not in _tables(con):
            return []
        cols = _columns(con, "analysed_time")
        if "reject_types" in cols:
            found = {str(r[0]) for r in con.execute(
                "SELECT DISTINCT reject_types FROM analysed_time") if r[0] is not None}
            odd = sorted(k for k in found if k != reject_key(k))
            if odd:
                raise ValueError(
                    f"{path}: analysed_time stores reject set(s) "
                    f"{'; '.join(repr(k) for k in odd)} that are not in TurtleWave's key "
                    "spelling (names sorted, comma-joined, no spaces, e.g. 'Arousal,Artefact'), "
                    "so they were written outside TurtleWave and TurtleWave's density would not "
                    "find them. Fix the database; the reader does not guess which set they mean.")
        elif {"reject_artifacts", "reject_arousals"} <= cols:
            found = {reject_key([n for n, on in (("Artefact", a), ("Arousal", b)) if on])
                     for a, b in con.execute("SELECT DISTINCT reject_artifacts, "
                                             "reject_arousals FROM analysed_time")}
        else:
            found = set()
    return sorted(found)


def _reject_sets(recordings, reject_types) -> list[str]:
    """The reject set each recording's density is read under: the configured one, else the
    one set its database stores. Refused before TurtleWave runs when a database stores none,
    several (and none is configured) or not the configured one, and when the recordings of one
    table would use different sets."""
    want = None if reject_types is None else reject_key(reject_types)
    keys = []
    for r in recordings:
        stored = stored_reject_sets(r.path)
        listed = "; ".join(reject_label(k) for k in stored)
        if not stored:
            raise ValueError(f"{r.path}: analysed_time stores no time under any reject set, "
                             "so there is nothing to divide the event count by and no density.")
        if want is None and len(stored) > 1:
            raise ValueError(
                f"{r.path} stores analysed time for {len(stored)} reject sets ({listed}). "
                "Their densities cover different seconds; name the one to use with "
                "reject_types in the [turtlewave] section.")
        if want is not None and want not in stored:
            raise ValueError(f"{r.path} stores analysed time only for the reject set(s) "
                             f"{listed}, not for {reject_label(want)} (reject_types).")
        keys.append(stored[0] if want is None else want)
    groups: dict[str, list[str]] = {}
    for r, k in zip(recordings, keys):
        groups.setdefault(k, []).append(f"{r.subject} {r.session}".strip())
    if len(groups) > 1:
        listed = "; ".join(f"{reject_label(k)}: {', '.join(w)}" for k, w in groups.items())
        raise ValueError(
            "The databases were detected with different reject sets, so their densities are "
            f"divided by time cleaned in different ways ({listed}) and are not comparable. "
            "Name one set every database stores with reject_types in the [turtlewave] section, "
            "or redetect so the runs match.")
    return keys


_STAGES = ("NREM1", "NREM2", "NREM3", "Wake", "REM")    # TurtleWave's stage vocabulary


def _stage_parts(stage) -> list[str]:
    """``['NREM2', 'NREM3']``, ``'NREM2'`` or the joint token ``'NREM2NREM3'`` as a list of
    stages; a token outside the vocabulary (``'Undefined'``) is one part. Mirrors
    TurtleWave's ``stage_components``."""
    if stage is None:
        return []
    if not isinstance(stage, str):
        return [str(s) for s in stage]
    parts, rest = [], stage
    while rest:
        hit = next((t for t in _STAGES if rest.startswith(t)), None)
        if hit is None:
            return [stage]
        parts.append(hit)
        rest = rest[len(hit):]
    return parts


def _covered_tokens(stored, stage) -> list[str]:
    """The stored stage tokens whose stages all lie inside the request, as TurtleWave's
    ``stage_tokens_covering`` does for density: ['NREM2', 'NREM3'] covers a joint
    'NREM2NREM3' run and per-stage 'NREM2' and 'NREM3' runs alike; 'NREM2' alone does not
    cover 'NREM2NREM3', whose events cannot be split by stage."""
    want = set(_stage_parts(stage))
    return [t for t in stored if t is not None and set(_stage_parts(t)) <= want]


def _run_scope(con, rec, event_type, method, stage, freq_lower, freq_upper):
    """The WHERE clause selecting the one detection run a request names, and the stored
    ``(method, stage, freq_lower, freq_upper)`` rows it covers (empty when no event matches).
    Two runs covering the request are refused, never pooled: they re-detect the same night."""
    where, args = ["event_type = ?"], [event_type]
    for name, value in (("method", method), ("freq_lower", freq_lower),
                        ("freq_upper", freq_upper)):
        if value is not None:
            where.append(f"{name} = ?")
            args.append(value)
    if stage is not None:
        stored = [r[0] for r in con.execute(
            f"SELECT DISTINCT stage FROM events WHERE {' AND '.join(where)}", args)]
        tokens = _covered_tokens(stored, stage)
        where.append(f"stage IN ({', '.join('?' * len(tokens))})" if tokens else "0")
        args += tokens
    scope = con.execute("SELECT DISTINCT method, stage, freq_lower, freq_upper FROM events "
                        f"WHERE {' AND '.join(where)}", args).fetchall()
    runs = {(m, lo, hi) for m, _, lo, hi in scope}
    parts = [p for _, t, _, _ in scope for p in _stage_parts(t)]
    if len(runs) > 1 or (stage is None and len(scope) > 1) or len(parts) > len(set(parts)):
        raise ValueError(
            f"{rec.path}: the request matches more than one method, stage or band "
            f"({scope}); name method, stage, freq_lower and freq_upper so one is selected. "
            "Stage tokens are pooled only when stage names them and no stage is in two.")
    return where, args, scope


def _sql_measure(rec, measure, event_type, method, stage, freq_lower, freq_upper) -> pd.Series:
    column = {"amplitude": "AVG(peak2peak_amp)", "duration": "AVG(duration)",
              "count": "COUNT(*)"}[measure]
    with sqlite3.connect(f"file:{rec.path}?mode=ro", uri=True) as con:
        where, args, _ = _run_scope(con, rec, event_type, method, stage, freq_lower, freq_upper)
        rows = con.execute(f"SELECT channel, {column} FROM events WHERE {' AND '.join(where)} "
                           "GROUP BY channel", args).fetchall()
    return pd.Series(dict(rows), dtype=float)


def _strict_parts(token) -> list[str] | None:
    """Stages of a token, or None when it is not made of known stages (TurtleWave's
    ``split_stage_token`` raises there and ``_run_stage_components`` skips the token)."""
    parts = _stage_parts(token)
    return parts if all(p in _STAGES for p in parts) else None


def _method_matches(stored, wanted) -> bool:
    """A run's stored method set ('_'-joined for a multi-method run) covers the method asked
    for. Mirrors TurtleWave's ``density._method_matches``."""
    return wanted is None or str(stored) == str(wanted) or str(wanted) in str(stored).split("_")


def _searched_stages(con, event_type, method, freq_lower, freq_upper) -> list[str]:
    """The stages the selected run searched, recovered as TurtleWave's
    ``density._run_stage_components`` does: processing_status for this event type, method and
    band, else detection_runs.stages, else the stage tokens its events are stored under; joint
    tokens expand to their stages. Empty when none of the three says."""
    tables, found = _tables(con), set()
    if "processing_status" in tables:
        where, args = ["success = 1", "event_type = ?"], [event_type]
        for name, value in (("freq_lower", freq_lower), ("freq_upper", freq_upper)):
            if value is not None:
                where.append(f"{name} = ?")
                args.append(float(value))
        for m, token in con.execute("SELECT DISTINCT method, stage FROM processing_status "
                                    f"WHERE {' AND '.join(where)}", args):
            if token and _method_matches(m, method):
                found.update(_strict_parts(token) or [])
    if not found and "detection_runs" in tables and \
            {"event_type", "method", "stages"} <= _columns(con, "detection_runs"):
        for m, cell in con.execute("SELECT method, stages FROM detection_runs "
                                   "WHERE event_type = ?", [event_type]):
            if not cell or not _method_matches(m, method):
                continue
            try:
                value = ast.literal_eval(cell)
            except (ValueError, SyntaxError):
                value = cell
            if isinstance(value, (list, tuple)):
                found.update(str(v) for v in value)
            elif isinstance(value, str):
                found.update(_strict_parts(value) or [])
    if not found:
        where, args = ["event_type = ?"], [event_type]
        for name, value in (("method", method), ("freq_lower", freq_lower),
                            ("freq_upper", freq_upper)):
            if value is not None:
                where.append(f"{name} = ?")
                args.append(value)
        for (token,) in con.execute(f"SELECT DISTINCT stage FROM events "
                                    f"WHERE {' AND '.join(where)}", args):
            found.update(_stage_parts(token))
    return _in_order(found)


def _in_order(stages) -> list[str]:
    order = ("NREM1", "NREM2", "NREM3", "REM", "Wake")       # TurtleWave's canonical order
    return sorted(set(stages), key=lambda s: (order.index(s) if s in order else len(order), s))


def _density_scope(rec, event_type, method, stage, freq_lower, freq_upper):
    """The selected run's method and band, and the stages its density is divided by: the
    requested stages the run actually searched. A slow-wave run over NREM3 alone, asked for
    NREM2 and NREM3, is divided by NREM3 time, not by the NREM2 time a spindle run on the same
    night put into analysed_time. Plain SQL, so a mixed set is refused before TurtleWave runs."""
    with sqlite3.connect(f"file:{rec.path}?mode=ro", uri=True) as con:
        if "analysed_time" not in _tables(con):
            raise ValueError(f"{rec.path} stores no analysed time (it predates TurtleWave 4.3), "
                             "so it has no density. Amplitude, duration and count still work.")
        _, _, scope = _run_scope(con, rec, event_type, method, stage, freq_lower, freq_upper)
        if scope:
            # The selected run names the method and band, so TurtleWave's roster of channels
            # that ran and found nothing (processing_status) is that run's, not every method's.
            method, _, freq_lower, freq_upper = scope[0]
        searched = _searched_stages(con, event_type, method, freq_lower, freq_upper)
    if stage is None:
        return method, freq_lower, freq_upper, searched
    requested = _in_order(_stage_parts(stage))
    used = [s for s in requested if s in searched] if searched else requested
    if not used:
        raise ValueError(f"{rec.path}: the selected '{event_type}' run searched "
                         f"{'+'.join(searched)}, none of the requested stages "
                         f"({'+'.join(requested)}), so it has no density for them.")
    return method, freq_lower, freq_upper, used


def _density(rec, event_type, method, stage, freq_lower, freq_upper,
             reject: str) -> pd.Series:
    """TurtleWave's density for the scope ``_density_scope`` resolved; ``stage`` is the list of
    stages to divide by (None leaves the scope to TurtleWave). ``reject`` is the reject set
    (``reject_key`` text) passed to TurtleWave explicitly, never left to its default."""
    try:
        from turtlewave_hdEEG.density import event_density
    except ImportError as err:
        raise ImportError("Density uses TurtleWave's own event_density; install "
                          "turtlewave-hdeeg (pip install hdeeg-snpm[turtlewave]).") from err
    d = event_density(str(rec.path), event_type=event_type, method=method, stage=stage,
                      freq_lower=freq_lower, freq_upper=freq_upper,
                      combine_stages=stage is not None and len(stage) > 1,
                      reject_types=[n for n in reject.split(",") if n])
    if d["stage"].nunique() > 1 or d["method"].nunique() > 1:
        raise ValueError(f"{rec.path}: density came back for more than one method or stage "
                         f"({sorted(set(zip(d['method'], d['stage'])))}); no event in scope "
                         "selects a run, so name method and stage.")
    return pd.Series(d["density_per_min"].to_numpy(float), index=d["channel"].to_numpy())


def event_table(recordings, measure: str, event_type: str, *, method=None, stage=None,
                freq_lower=None, freq_upper=None, drop_interpolated: bool = True,
                reject_types=None):
    """One row per recording, one column per channel.

    Returns ``(table, manifest)``. ``table`` has ``Subject`` and ``session`` then the channels;
    a channel with no event is missing for amplitude and duration and 0 for count and density.
    ``manifest`` lists, per recording, the database path, where the subject ID came from and
    the interpolated channels; for density also ``stages``, the stages whose analysed
    time each density was divided by (the requested stages its run searched; subjects that
    differ are refused) and ``reject_types``, the reject set whose analysed time it was divided
    by (``none`` for the set that rejects nothing). That set is ``reject_types`` when given
    (text such as ``"Artefact"`` or a list of names; ``""`` or ``"none"`` rejects nothing), else the one set the database stores, which is the set
    detection used; recordings using different sets are refused. With ``drop_interpolated``
    those channels are set to missing for that recording, so the analysis reports them as
    excluded or tests them on fewer subjects.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    if measure == "density":
        scopes = [_density_scope(r, event_type, method, stage, freq_lower, freq_upper)
                  for r in recordings]
        used = ["+".join(sc[3]) for sc in scopes]
        _check_stages(recordings, used, stage)
        rejects = _reject_sets(recordings, reject_types)
    rows, manifest = [], []
    for i, r in enumerate(recordings):
        if measure == "density":
            m, lo, hi, stages = scopes[i]
            values = _density(r, event_type, m, None if stage is None else stages, lo, hi,
                              rejects[i])
        else:
            values = _sql_measure(r, measure, event_type, method, stage, freq_lower, freq_upper)
        if values.empty:
            raise ValueError(f"{r.path}: no '{event_type}' events match the request. "
                             "Use available() to see what the database holds.")
        interp = interpolated_channels(r.path)
        if drop_interpolated:
            values[[c for c in interp if c in values.index]] = np.nan
        rows.append(values)
        manifest.append({"Subject": r.subject, "session": r.session, "database": str(r.path),
                         "subject_source": r.subject_source,
                         "n_channels_with_events": int(values.notna().sum()),
                         "interpolated_channels": " ".join(interp)})
    if measure == "density":
        for row, u, k in zip(manifest, used, rejects):
            row["stages"] = u
            row["reject_types"] = reject_written(k)
    wide = pd.DataFrame(rows).reset_index(drop=True)
    wide = wide[sorted(wide.columns, key=_natural)]
    if measure == "count":
        wide = wide.fillna(0)
        if drop_interpolated:            # zero means "no event", not "channel not usable"
            for i, row in enumerate(manifest):
                hit = [c for c in row["interpolated_channels"].split() if c in wide.columns]
                wide.loc[i, hit] = np.nan
    wide.insert(0, "session", [r.session for r in recordings])
    wide.insert(0, "Subject", [r.subject for r in recordings])
    return wide, pd.DataFrame(manifest)


def _check_stages(recordings, used, stage):
    """One table, one denominator: refuse subjects whose densities cover different stages, and
    warn when every subject's run searched fewer stages than were requested."""
    def who(r):
        return f"{r.subject} {r.session}".strip()
    groups: dict[str, list[str]] = {}
    for r, u in zip(recordings, used):
        groups.setdefault(u, []).append(who(r))
    if len(groups) > 1:
        listed = "; ".join(f"{u or 'none'}: {', '.join(subs)}" for u, subs in groups.items())
        raise ValueError(
            "Density would be divided by the time of different stages for different subjects, "
            f"because their runs searched different stages ({listed}). Values over different "
            "stages are not comparable; request only the stages every run searched, or redetect "
            "so the runs match.")
    if stage is not None and used:
        requested = "+".join(_in_order(_stage_parts(stage)))
        if used[0] != requested:
            warnings.warn(
                f"Density was requested over {requested} but the selected runs searched only "
                f"{used[0]}, so it is divided by {used[0]} time for every subject "
                f"({', '.join(groups[used[0]])}). See the stages column of recordings.csv.",
                stacklevel=3)


def _natural(label: str):
    m = re.match(r"^E(\d+)$", str(label))
    return (0, int(m.group(1)), "") if m else (1, 0, str(label))


def write_tables(source, out_dir, event_type: str, measures=("density", "amplitude", "duration"),
                 subject_from: str = "auto", **scope) -> dict[str, Path]:
    """Write ``eventStat_<measure>_<event_type>.csv`` for each measure and ``recordings.csv``
    (the resolved subject, session and path of every database, to be checked by eye)."""
    recordings = find_recordings(source, subject_from)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths, manifest = {}, None
    for m in measures:
        table, listed = event_table(recordings, m, event_type, **scope)
        if manifest is None or m == "density":      # keep the stages column
            manifest = listed
        paths[m] = out_dir / f"eventStat_{m}_{event_type}.csv"
        table.to_csv(paths[m], index=False)
    paths["recordings"] = out_dir / "recordings.csv"
    manifest.to_csv(paths["recordings"], index=False)
    return paths
