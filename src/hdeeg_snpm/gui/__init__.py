"""A window that runs an analysis straight from TurtleWave databases (``snpm gui``).

The window needs PyQt5 (``pip install hdeeg-snpm[gui]``); nothing else in the package does, so
PyQt5 is imported only in ``window.py``.
"""

from __future__ import annotations

import sys

INSTALL_HINT = ("The window needs PyQt5, which is not installed. Install it with: "
                "pip install \"hdeeg-snpm[gui]\"")


def main(argv=None) -> int:
    """Open the window; return 1 with an install hint when PyQt5 is missing."""
    try:
        from . import window
    except ModuleNotFoundError as err:
        if not (err.name or "").startswith("PyQt5"):
            raise
        print(f"snpm: {INSTALL_HINT}", file=sys.stderr)
        return 1
    return window.main(argv)
