"""Convert the MATLAB montage assets to the .npz files shipped in hdeeg_snpm/assets.

Run once, from anywhere:

    python tools/convert_matlab_assets.py /path/to/SnPM_2025/dependencies

Each output holds the channel labels, their EEGLAB polar position (theta in degrees, radius),
Cartesian position, and the neighbour matrix converted from MATLAB's 1-based, NaN-padded form to
0-based integers padded with -1. Channel order is the order of the MATLAB chanlocs struct, which
is the order the neighbour matrix indexes into.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

SYSTEMS = {
    "egi": ("EEG178chanlocs", "EEG178chanlocs", "NeighborMatrix_178"),
    "compu": ("compu257_chanlocs", "compu257_chanlocs", "NeighborMatrix_compu257"),
}


def convert(dep_dir: Path, out_dir: Path) -> None:
    for key, (loc_file, loc_var, nb_file) in SYSTEMS.items():
        locs = sio.loadmat(dep_dir / f"{loc_file}.mat", simplify_cells=True,
                           variable_names=[loc_var])[loc_var]
        nb = sio.loadmat(dep_dir / f"{nb_file}.mat")["neighbors"]
        if nb.shape[0] != len(locs):
            raise ValueError(f"{key}: {len(locs)} chanlocs but {nb.shape[0]} neighbour rows")
        neighbors = np.where(np.isnan(nb), -1, nb - 1).astype(np.int32)
        if neighbors.max() >= len(locs) or (neighbors < -1).any():
            raise ValueError(f"{key}: neighbour index out of range")
        np.savez(
            out_dir / f"{key}.npz",
            labels=np.array([str(c["labels"]) for c in locs]),
            theta=np.array([float(c["theta"]) for c in locs]),
            radius=np.array([float(c["radius"]) for c in locs]),
            xyz=np.array([[float(c["X"]), float(c["Y"]), float(c["Z"])] for c in locs]),
            neighbors=neighbors,
        )
        print(f"{key}: {len(locs)} channels, neighbour matrix {neighbors.shape}")


if __name__ == "__main__":
    dep = Path(sys.argv[1])
    out = Path(__file__).resolve().parents[1] / "src" / "hdeeg_snpm" / "assets"
    convert(dep, out)
