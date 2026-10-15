"""Read TurtleWave-hdEEG ``neural_events.db`` files into subjects-by-channels tables.

TurtleWave writes one database per recording. This module finds the databases, works out which
subject and session each belongs to, and builds one wide table per measure (density, amplitude,
duration or count) for one event type, method, stage and frequency band.

Amplitude, duration and count are plain SQL. Density is delegated to TurtleWave's own
``event_density`` (install with ``pip install hdeeg-snpm[turtlewave]``), so the rule for the
time an event count is divided by lives in one place.
"""

from __future__ import annotations

import json
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
    prefix) or the file is named in an error. ``'database'`` or ``'folder'`` forces one source,
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
        folder_subject, session = _folder_ids(db.resolve())
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


def _stage_token(stage) -> str | None:
    if stage is None or isinstance(stage, str):
        return stage
    return "".join(stage)


def _sql_measure(rec, measure, event_type, method, stage, freq_lower, freq_upper) -> pd.Series:
    column = {"amplitude": "AVG(peak2peak_amp)", "duration": "AVG(duration)",
              "count": "COUNT(*)"}[measure]
    where, args = ["event_type = ?"], [event_type]
    for name, value in (("method", method), ("stage", _stage_token(stage)),
                        ("freq_lower", freq_lower), ("freq_upper", freq_upper)):
        if value is not None:
            where.append(f"{name} = ?")
            args.append(value)
    with sqlite3.connect(f"file:{rec.path}?mode=ro", uri=True) as con:
        scope = con.execute("SELECT DISTINCT method, stage, freq_lower, freq_upper FROM events "
                            f"WHERE {' AND '.join(where)}", args).fetchall()
        if len(scope) > 1:
            raise ValueError(
                f"{rec.path}: the request matches more than one method, stage or band "
                f"({scope}); name method, stage, freq_lower and freq_upper so one is selected.")
        rows = con.execute(f"SELECT channel, {column} FROM events WHERE {' AND '.join(where)} "
                           "GROUP BY channel", args).fetchall()
    return pd.Series(dict(rows), dtype=float)


def _density(rec, event_type, method, stage, freq_lower, freq_upper) -> pd.Series:
    try:
        from turtlewave_hdEEG.density import event_density
    except ImportError as err:
        raise ImportError("Density uses TurtleWave's own event_density; install "
                          "turtlewave-hdeeg (pip install hdeeg-snpm[turtlewave]).") from err
    d = event_density(str(rec.path), event_type=event_type, method=method, stage=stage,
                      freq_lower=freq_lower, freq_upper=freq_upper,
                      combine_stages=not (stage is None or isinstance(stage, str)))
    if d["stage"].nunique() > 1 or d["method"].nunique() > 1:
        raise ValueError(f"{rec.path}: density came back for more than one method or stage "
                         f"({sorted(set(zip(d['method'], d['stage'])))}); name one.")
    return pd.Series(d["density_per_min"].to_numpy(float), index=d["channel"].to_numpy())


def event_table(recordings, measure: str, event_type: str, *, method=None, stage=None,
                freq_lower=None, freq_upper=None, drop_interpolated: bool = True):
    """One row per recording, one column per channel.

    Returns ``(table, manifest)``. ``table`` has ``Subject`` and ``session`` then the channels;
    a channel with no event is missing for amplitude and duration and 0 for count and density.
    ``manifest`` lists, per recording, the database path, where the subject ID came from and
    the interpolated channels. With ``drop_interpolated`` those channels are set to missing for
    that recording, so the analysis reports them as excluded or tests them on fewer subjects.
    """
    if measure not in MEASURES:
        raise ValueError(f"measure must be one of {MEASURES}")
    rows, manifest = [], []
    for r in recordings:
        if measure == "density":
            values = _density(r, event_type, method, stage, freq_lower, freq_upper)
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
        table, manifest = event_table(recordings, m, event_type, **scope)
        paths[m] = out_dir / f"eventStat_{m}_{event_type}.csv"
        table.to_csv(paths[m], index=False)
    paths["recordings"] = out_dir / "recordings.csv"
    manifest.to_csv(paths["recordings"], index=False)
    return paths
