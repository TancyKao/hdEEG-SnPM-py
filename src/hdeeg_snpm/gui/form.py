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
        p["turtlewave"] = section
        return p

    def to_toml(self) -> str:
        """The run config as TOML text, for ``snpm run``."""
        params = self.to_params()
        section = params.pop("turtlewave")
        lines = ["# Written by the hdeeg-snpm window. Run it again with: snpm run <this file>"]
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
