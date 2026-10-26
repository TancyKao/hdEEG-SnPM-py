"""What the GUI window holds, without Qt: the choices made so far, turned into the parameters
of a ``[turtlewave]`` run config.

``FormState.to_params()`` gives the dictionary ``cli.load_config`` would give for the same
config file, and ``to_toml()`` writes that file. The window only fills a ``FormState``; every
check of the request is left to ``io.turtlewave_config.prepare`` and the pipeline.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..io import tables
from ..io import turtlewave as tw
from ..io.turtlewave_config import read_subjects

ANALYSES = {
    "pairedT": "Paired t-test (two sessions)",
    "unpairedT": "Unpaired t-test (two groups)",
    "anova1": "One-way ANOVA (groups)",
    "correlationP": "Correlation, Pearson",
    "correlationS": "Correlation, Spearman",
}
T_TESTS = ("pairedT", "unpairedT")


def density_available() -> bool:
    """Density is computed by TurtleWave itself, so it needs TurtleWave installed."""
    return importlib.util.find_spec("turtlewave_hdEEG") is not None


def choices(available: pd.DataFrame, event_type: str) -> dict:
    """Methods, stages and bands the databases hold for one event type (from
    ``turtlewave.available``). Joint stage tokens such as ``NREM2NREM3`` are listed by stage."""
    rows = available[available["event_type"] == event_type]
    stages = [p for token in rows["stage"].dropna().unique() for p in tw._stage_parts(token)]
    bands = sorted({(float(lo), float(hi)) for lo, hi in zip(rows["freq_lower"],
                                                             rows["freq_upper"])
                    if pd.notna(lo) and pd.notna(hi)})     # a run stored without a band
    return {"methods": sorted(rows["method"].astype(str).unique()),
            "stages": tw._in_order(stages), "bands": bands}


NO_TIME = "no analysed time stored, so no density"


def _stored_sets(recordings) -> list[tuple[str, tuple | None, str]]:
    """(who, reject sets stored or None when unreadable, error text) per database."""
    out = []
    for r in recordings:
        try:
            out.append((f"{r.subject} {r.session}".strip(),
                        tuple(tw.stored_reject_sets(r.path)), ""))
        except Exception as err:  # noqa: BLE001 - shown; the run reports it again
            out.append((f"{r.subject} {r.session}".strip(), None, str(err)))
    return out


def reject_choices(recordings) -> list[str]:
    """The reject sets to choose from when a database stores more than one: the sets every
    database stores. Empty when there is nothing to choose (each stores one set) or no choice
    could serve every database."""
    found = [sets for _, sets, _ in _stored_sets(recordings)]
    if not found or any(s is None for s in found) or all(len(s) <= 1 for s in found):
        return []
    return sorted(set.intersection(*(set(s) for s in found)))


def reject_unshared(recordings) -> bool:
    """True when every database stores analysed time but no reject set is stored by all of
    them: no choice can make their densities comparable, so the window does not run."""
    found = [sets for _, sets, _ in _stored_sets(recordings)]
    return bool(found) and all(found) and \
        not set.intersection(*(set(s) for s in found))


def reject_text(recordings) -> str:
    """What density's denominator was cleaned of, read from the databases: the one reject set
    they all store (``"Artefact (as detected)"``), or what differs. Density uses that set; a
    database without one is refused when the run starts, and one with several needs a choice."""
    found = _stored_sets(recordings)
    if found and all(sets is None for _, sets, _ in found):
        return "could not read the reject sets of the databases: " + found[0][2]

    def describe(sets):
        if sets is None:
            return "unreadable"
        if not sets:
            return NO_TIME
        if len(sets) > 1:
            return "several reject sets stored: " + " and ".join(map(tw.reject_label, sets))
        return tw.reject_label(sets[0])
    groups: dict = {}
    for who, sets, _ in found:
        groups.setdefault(sets, []).append(who)
    if reject_unshared(recordings):
        return ("no reject set is stored by every database ("
                + "; ".join(f"{describe(k)}: {', '.join(who)}" for k, who in groups.items())
                + "); these densities cannot be compared. Redetect so the runs match.")
    if len(groups) == 1:
        (sets,) = groups
        if sets is not None and len(sets) == 1:
            return f"{describe(sets)} (as detected)"
        return describe(sets) + ("" if sets == () else f" ({', '.join(groups[sets])})")
    return "differs between databases: " + "; ".join(
        f"{describe(k)} ({', '.join(who)})" for k, who in groups.items())


_CLI_HINTS = ((("Name one set every database stores with reject_types in the [turtlewave] "
               "section, or redetect so the runs match."),
               ("No reject set is stored by every database, so these densities cannot be "
               "compared. Redetect so the runs match.")),
              ("pass subject_from='database' or 'folder' to say which is right",
               "set Subject ID from to folder or database"),
              ("use subject_from='folder'", "set Subject ID from to folder"),
              ("with reject_types in the [turtlewave] section",
               'in "Rejected before density"'))


def window_message(text: str) -> str:
    """A loader error worded for the window, which sets the subject-ID source and the reject
    set with menus."""
    for cli, window in _CLI_HINTS:
        text = text.replace(cli, window)
    return text


def subjects_table(path) -> pd.DataFrame:
    """A subjects file read as text, as ``prepare`` compares group labels, for listing its
    columns and values in the window."""
    return read_subjects(path, as_text=True)[0]


def info_columns(subjects: pd.DataFrame) -> list[str]:
    """Columns of a subjects file other than its subject column."""
    sid = tables.subject_column(subjects.columns)
    return [str(c) for c in subjects.columns if c != sid]


@dataclass
class FormState:
    folder: str = ""                                          # the folder read, not the text
    subject_from: str = "auto"
    sessions_found: list[str] = field(default_factory=list)   # from the databases, not a param
    stages_found: list[str] = field(default_factory=list)     # from the databases, not a param
    reject_sets_found: list[str] = field(default_factory=list)   # a choice is needed; not a param
    reject_types: str | None = None        # density: the reject set chosen ('' rejects nothing)
    reject_unshared: bool = False          # no set stored by every database; not a param
    event_type: str = ""
    method: str = ""
    stage: list[str] = field(default_factory=list)
    band: tuple[float, float] | None = None
    measure: str = "amplitude"
    comparison: str = "unpairedT"
    tail: str = "both"
    permutations: int = 5000
    seed: int | None = None
    channels: str = "egi"
    datatype: str = "absolute"
    sessions: list[str | None] = field(default_factory=list)   # pairedT: the two sessions
    session: str | None = None             # one-recording analyses; '' is "no session folder"
    subjects_file: str = ""
    group_col: str = ""
    groups: list[str] = field(default_factory=list)
    behavior_col: str = ""
    output_path: str = ""

    def needs_session(self) -> bool:
        return self.comparison != "pairedT" and len(set(self.sessions_found)) > 1

    def roles(self) -> list[str]:
        """The role inputs the chosen analysis shows."""
        shown = {"pairedT": ["sessions"],
                 "unpairedT": ["subjects_file", "group_col", "groups"],
                 "anova1": ["subjects_file", "group_col"],
                 "correlationP": ["subjects_file", "behavior_col"],
                 "correlationS": ["subjects_file", "behavior_col"]}[self.comparison]
        return shown + (["session"] if self.needs_session() else [])

    def missing(self) -> list[str]:
        """What still has to be filled before the analysis can run."""
        todo = []
        if not self.folder:
            todo.append("a study folder with databases (press Enter or Load to read it)")
        if not self.event_type:
            todo.append("an event type")
        if self.stages_found and not self.stage:
            todo.append("at least one stage")
        if not self.measure:
            todo.append("a measure")
        if self.measure == "density" and self.reject_sets_found and self.reject_types is None:
            todo.append("the reject set of density")
        if self.measure == "density" and self.reject_unshared:
            todo.append("databases that share a reject set (see Rejected before density)")
        roles = self.roles()
        if "sessions" in roles and (len(self.sessions) != 2 or None in self.sessions
                                    or self.sessions[0] == self.sessions[1]):
            todo.append("two different sessions")
        if "subjects_file" in roles and not self.subjects_file:
            todo.append("a subjects file (press Enter or Load to read it)")
        if "group_col" in roles and not self.group_col:
            todo.append("a group column")
        if "groups" in roles and (len(self.groups) != 2 or not all(self.groups)
                                  or self.groups[0] == self.groups[1]):
            todo.append("two different groups")
        if "behavior_col" in roles and not self.behavior_col:
            todo.append("a score column")
        if "session" in roles and self.session is None:
            todo.append("the session to analyse")
        if not self.output_path:
            todo.append("an output folder")
        return todo

    def to_params(self) -> dict:
        """The parameters ``cli.load_config`` returns for the config ``to_toml`` writes."""
        p = {"comparison": self.comparison}
        if self.comparison in T_TESTS:
            p["tail"] = self.tail
        p.update(channels=self.channels, datatype=self.datatype,
                 permutations=int(self.permutations))
        if self.seed is not None:
            p["seed"] = int(self.seed)
        roles = self.roles()
        if "sessions" in roles:
            p["sessions"] = list(self.sessions)
        if "session" in roles and self.session is not None:
            p["sessions"] = [self.session]
        if "group_col" in roles:
            p["group_col"] = self.group_col
        if "groups" in roles:
            p["groups"] = list(self.groups)
        if "behavior_col" in roles:
            p["behavior_col"] = self.behavior_col
        p["output_path"] = _absolute(self.output_path)
        section = {"folder": _absolute(self.folder)}
        if "subjects_file" in roles:
            section["subjects_file"] = _absolute(self.subjects_file)
        section["event_type"] = self.event_type
        if self.method:
            section["method"] = self.method
        if self.stage:
            section["stage"] = list(self.stage)
        if self.band is not None:
            section["freq_lower"], section["freq_upper"] = self.band
        section.update(measure=self.measure, subject_from=self.subject_from)
        if self.measure == "density" and self.reject_sets_found and self.reject_types is not None:
            section["reject_types"] = tw.reject_written(self.reject_types)
        p["turtlewave"] = section
        return p

    def to_toml(self) -> str:
        """The run config as TOML text, for ``snpm run``."""
        params = self.to_params()
        section = params.pop("turtlewave")
        lines = ["# Written by the hdEEG-SnPM window. Run it again with: snpm run <this file>"]
        lines += [f"{k} = {_value(v)}" for k, v in params.items()]
        lines += ["", "[turtlewave]"] + [f"{k} = {_value(v)}" for k, v in section.items()]
        return "\n".join(lines) + "\n"


def _absolute(path) -> str:
    return str(Path(path).resolve()) if path else ""


def _value(v) -> str:
    """One TOML value: text, a whole or finite decimal number, or a list of them. Text is a
    basic string: quote, backslash and control characters escaped, everything else (any
    Unicode character) written as itself in the UTF-8 file."""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_value(x) for x in v) + "]"
    if isinstance(v, str):
        return '"' + "".join(_char(c) for c in v) + '"'
    if isinstance(v, float) and math.isfinite(v):
        return repr(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    raise ValueError(f"cannot write {v!r} to the config")


def _char(c: str) -> str:
    if c in '"\\':
        return "\\" + c
    if 0xD800 <= ord(c) <= 0xDFFF:
        raise ValueError("a path holds bytes that are not valid text; rename it")
    if ord(c) < 0x20 or ord(c) == 0x7F:
        return f"\\u{ord(c):04X}"
    return c
