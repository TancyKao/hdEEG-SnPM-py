"""Make docs/images/gui-window.png: the window on the synthetic example study, ready to run.

Builds the synthetic databases with examples/make_turtlewave_example.py, fills the window
offscreen, and shows neutral folder names instead of this machine's paths.

    QT_QPA_PLATFORM=offscreen python tools/gui_screenshot.py
"""

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "example_output" / "turtlewave_study"


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from hdeeg_snpm.gui.window import MainWindow

    runpy.run_path(str(ROOT / "examples" / "make_turtlewave_example.py"), run_name="__main__")
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.resize(1100, 760)
    win.load_folder(STUDY / "databases")
    win.load_subjects(STUDY / "subjects.csv")
    win.out_edit.setText(str(ROOT / "example_output"))
    win.seed.setText("20261025")
    win.permutations.setValue(1000)
    # Neutral names in the picture; the state behind them is the real one.
    for edit, shown in ((win.folder_edit, "/data/study/databases"),
                        (win.subjects_edit, "/data/study/subjects.csv"),
                        (win.out_edit, "/data/results")):
        edit.blockSignals(True)
        edit.setText(shown)
        edit.blockSignals(False)
    win.show()
    app.processEvents()
    out = ROOT / "docs" / "images" / "gui-window.png"
    out.parent.mkdir(exist_ok=True)
    win.grab().save(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
