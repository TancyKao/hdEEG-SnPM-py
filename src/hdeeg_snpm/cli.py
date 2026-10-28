"""Command line: ``snpm run config.toml``, and ``snpm gui`` for the window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from . import __version__

_PATH_KEYS = ("data1_file", "data2_file", "data_file", "covariate_file", "count1_file",
              "count2_file", "output_path")


def load_config(path) -> dict:
    """Read a TOML file of parameters. Relative paths, including the ``folder`` and
    ``subjects_file`` of a ``[turtlewave]`` section, are taken from the file's own folder."""
    path = Path(path)
    with path.open("rb") as fh:
        params = tomllib.load(fh)
    for k in _PATH_KEYS:
        if params.get(k):
            params[k] = str((path.parent / params[k]).resolve())
    tw = params.get("turtlewave")
    if isinstance(tw, dict):
        for k in ("folder", "subjects_file"):
            if tw.get(k):
                tw[k] = str((path.parent / tw[k]).resolve())
    params.setdefault("output_path", str(path.parent / "snpm_output"))
    return params


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="snpm", description="Permutation statistics for "
                                     "high-density EEG scalp maps.")
    parser.add_argument("--version", action="version", version=f"hdEEG-SnPM {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the analysis described in a TOML file")
    run.add_argument("config")
    run.add_argument("--permutations", type=int, help="override the value in the file")
    run.add_argument("--seed", type=int, help="override the value in the file")
    sub.add_parser("gui", help="open the window that runs an analysis from TurtleWave "
                   "databases or a CSV/Excel table (needs PyQt5: pip install "
                   "\"hdeeg-snpm[gui]\")")
    args = parser.parse_args(argv)
    if args.command == "gui":
        from .gui import main as gui_main
        return gui_main([])

    from .output import run_and_write

    params = load_config(args.config)
    for k in ("permutations", "seed"):
        if getattr(args, k) is not None:
            params[k] = getattr(args, k)
    try:
        paths = run_and_write(params)
    except (ValueError, FileNotFoundError, KeyError, ImportError) as err:
        print(f"snpm: {err}", file=sys.stderr)
        return 1
    for kind, p in paths.items():
        print(f"{kind:5s} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
