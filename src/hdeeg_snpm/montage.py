"""Recording systems and label-based channel selection.

Ports ``snpm_montage_registry.m`` and ``snpm_setup_channels.m``. A recording system is defined by
one ``assets/<key>.npz`` file (built by ``tools/convert_matlab_assets.py``) plus an entry in
``_SYSTEMS``; that is the only place a system is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources

import numpy as np

from .clusters import ChannelGraph

_SYSTEMS = {
    "egi": ("EGI 256 (HydroCel)", ("164 channels", "178 channels", "egi257", "egi256")),
    "compu": ("Compumedics 257 (Neuvo)", ("compu257", "neuvo", "neuvo256", "compumedics")),
}


@dataclass(frozen=True)
class Montage:
    """The analysed scalp channels of one recording system, in asset order."""

    key: str
    display: str
    labels: tuple[str, ...]
    theta: np.ndarray       # EEGLAB polar angle, degrees
    radius: np.ndarray      # EEGLAB polar radius
    xyz: np.ndarray
    neighbors: np.ndarray   # (n, k) 0-based indices, -1 padded


@dataclass(frozen=True)
class ChannelSetup:
    """Data, channel positions and neighbour graph in one shared channel order."""

    system_key: str
    labels: tuple[str, ...]
    data: tuple[np.ndarray, ...]
    theta: np.ndarray
    radius: np.ndarray
    xyz: np.ndarray
    neighbors: np.ndarray
    graph: ChannelGraph
    data_columns: np.ndarray     # which input columns were kept
    montage_indices: np.ndarray  # their indices in the system's asset order


def resolve_key(key: str) -> str:
    """Return the system key for a key or legacy alias (case-insensitive)."""
    k = str(key).lower()
    for system, (_, aliases) in _SYSTEMS.items():
        if k == system or k in (a.lower() for a in aliases):
            return system
    raise ValueError(f'Unknown recording system "{key}". Known systems: {", ".join(_SYSTEMS)}.')


@cache
def get_montage(key: str) -> Montage:
    key = resolve_key(key)
    with resources.files("hdeeg_snpm.assets").joinpath(f"{key}.npz").open("rb") as f:
        z = np.load(f)
        return Montage(
            key=key,
            display=_SYSTEMS[key][0],
            labels=tuple(str(s) for s in z["labels"]),
            theta=z["theta"],
            radius=z["radius"],
            xyz=z["xyz"],
            neighbors=z["neighbors"],
        )


def remap_neighbors(neighbors: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Subset a neighbour matrix to channels ``sel`` and renumber entries to the new order.

    Neighbours that were not kept are dropped. Rows stay left-packed and -1 padded; columns that
    end up empty are trimmed.
    """
    sel = np.asarray(sel)
    remap = np.full(neighbors.shape[0], -1)
    remap[sel] = np.arange(sel.size)
    out = np.full((sel.size, neighbors.shape[1]), -1, dtype=neighbors.dtype)
    for i, row in enumerate(neighbors[sel]):
        kept = remap[row[row >= 0]]
        kept = kept[kept >= 0]
        out[i, :kept.size] = kept
    out = out[:, (out >= 0).any(axis=0)]
    return out if out.shape[1] else np.full((sel.size, 1), -1, dtype=neighbors.dtype)


@cache
def all_labels() -> frozenset[str]:
    """Channel labels of every supported recording system."""
    return frozenset(label for key in _SYSTEMS for label in get_montage(key).labels)


def setup_channels(key: str, data, channel_cols) -> ChannelSetup:
    """Keep the data columns whose names are channels of the recording system.

    Parameters
    ----------
    key : recording-system key or legacy alias.
    data : sequence of (rows, columns) arrays sharing one column order.
    channel_cols : the column names, in column order.

    Channels are kept in data-column order (not asset order); positions and the neighbour matrix
    are subset and renumbered to match. Matching is exact and case-sensitive.
    """
    m = get_montage(key)
    index = {label: i for i, label in enumerate(m.labels)}
    cols = [str(c) for c in channel_cols]
    data_columns = np.array([j for j, c in enumerate(cols) if c in index], dtype=int)
    if data_columns.size == 0:
        raise ValueError(f"No data columns match the {m.display} montage labels.")
    sel = np.array([index[cols[j]] for j in data_columns], dtype=int)
    if np.unique(sel).size != sel.size:
        raise ValueError("A channel label appears more than once in the data columns.")
    neighbors = remap_neighbors(m.neighbors, sel)
    return ChannelSetup(
        system_key=m.key,
        labels=tuple(cols[j] for j in data_columns),
        data=tuple(np.asarray(d)[:, data_columns] for d in data),
        theta=m.theta[sel],
        radius=m.radius[sel],
        xyz=m.xyz[sel],
        neighbors=neighbors,
        graph=ChannelGraph(neighbors),
        data_columns=data_columns,
        montage_indices=sel,
    )
