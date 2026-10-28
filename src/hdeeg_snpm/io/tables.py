"""Wide CSV / Excel tables: reading, subject IDs and recognising what a table holds.

Ports the table handling of ``core_snpm_analysis.m`` and ``snpm_detect_behavior_table.m``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..montage import all_labels

_EGI_NAME = re.compile(r"^E\d+$")
# pandas 3 parses every float exactly (to the nearest double) by default and deprecates
# float_precision; before 3 the default parser can be one unit in the last place off.
_EXACT = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}


def read_csv(path, **kwargs) -> pd.DataFrame:
    """``pandas.read_csv`` with every number parsed to the nearest double, as MATLAB reads it."""
    return pd.read_csv(path, **_EXACT, **kwargs)


def read_table(path, sheet=None) -> pd.DataFrame:
    """Read a CSV file, or one sheet of an Excel workbook (``sheet=None`` or ``'CSV File'`` reads
    CSV). Numbers are parsed so that the value equals the text exactly as MATLAB reads it.
    Column names are always text, so a header such as 1995 is the column ``'1995'`` wherever a
    column is named (``meta_cols``, roles); Excel would otherwise give it as a number."""
    path = Path(path)
    if sheet in (None, "", "CSV File") and path.suffix.lower() not in (".xlsx", ".xls", ".xlsm"):
        table = read_csv(path)
    else:
        table = pd.read_excel(path, sheet_name=0 if sheet in (None, "", "CSV File") else sheet)
    table.columns = [str(c) for c in table.columns]
    return table


def subject_column(columns) -> str | None:
    """The first column whose name contains 'subject' (any case); the only ID column removed."""
    return next((c for c in columns if "subject" in str(c).lower()), None)


def subject_ids(values) -> np.ndarray:
    """IDs as text. Numeric IDs become ``sub001``-style so 1 and 'sub001' match across files."""
    v = pd.Series(values)
    if pd.api.types.is_numeric_dtype(v):
        return np.array([f"sub{int(x):03d}" for x in v], dtype=object)
    return v.astype(str).to_numpy(dtype=object)


@dataclass(frozen=True)
class TableKind:
    kind: str                  # 'behavior', 'montage' or 'unknown'
    col_name: str              # the measure column of a behaviour table, else ''
    channel_cols: tuple[str, ...]
    nonchannel_cols: tuple[str, ...]
    data_cols: tuple[str, ...]  # every column except the subject ID


def detect_table(columns) -> TableKind:
    """Say whether a table is one behavioural score per subject or a subjects-by-channels table.

    A column counts as a channel if it is named like ``E12``, is ``Cz`` (any case), or is a label
    of any supported recording system. One data column that is not a channel is a behaviour
    table; more than half the data columns being channels is a montage table.
    """
    cols = [str(c) for c in columns]
    subj = subject_column(cols)
    known = all_labels()
    data = [c for c in cols if c != subj]
    if not data:
        raise ValueError(f"The table has no data column (columns: {', '.join(cols)}).")
    chan = [c for c in data if _EGI_NAME.match(c) or c.lower() == "cz" or c in known]
    other = [c for c in data if c not in chan]
    if len(data) == 1:
        kind = "montage" if chan else "behavior"
    else:
        kind = "montage" if len(chan) / len(data) > 0.5 else "unknown"
    return TableKind(kind, data[0] if kind == "behavior" else "", tuple(chan), tuple(other),
                     tuple(data))


def numeric_matrix(table: pd.DataFrame, which: str) -> np.ndarray:
    """The table as a float matrix; names the offending columns if any is not numeric."""
    bad = [str(c) for c in table.columns
           if not (pd.api.types.is_numeric_dtype(table[c]) or pd.api.types.is_bool_dtype(table[c]))]
    if bad:
        raise ValueError(
            f"{which} still has non-numeric column(s) after the subject ID was removed: "
            f"{', '.join(bad)}. Only the first column whose name contains 'subject' is removed; "
            "every other column is read as data and must hold numbers.")
    return table.to_numpy(dtype=float)
