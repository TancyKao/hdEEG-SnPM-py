"""Make the window pictures in docs/images, filled offscreen and ready to run:

- gui-window.png: the TurtleWave source on the synthetic example study (unpaired t-test);
- gui-window-table.png: the table source on a synthetic subjects-by-channels table with log AHI,
  age and sex (regression on log AHI adjusted for age and sex).

The databases come from examples/make_turtlewave_example.py and the table from the example
data in examples/data; the pictures show neutral folder names instead of this machine's paths.

    QT_QPA_PLATFORM=offscreen python tools/gui_screenshot.py
"""

import os
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "example_output" / "turtlewave_study"
TABLE = ROOT / "example_output" / "table_study" / "band_power.csv"


def make_table():
    """The paired example's first condition with synthetic log AHI, age and sex columns."""
    eeg = pd.read_csv(ROOT / "examples" / "data" / "paired_condA.csv")
    rng = np.random.default_rng(20261026)
    n = len(eeg)
    info = pd.DataFrame({"Subject": eeg["Subject"], "logAHI": np.round(rng.normal(1.2, 0.4, n), 3),
                         "age": rng.integers(30, 70, n), "sex": ["F", "M"] * (n // 2)})
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([info, eeg.drop(columns="Subject")], axis=1).to_csv(TABLE, index=False)


def neutral(pairs):
    for edit, shown in pairs:
        edit.blockSignals(True)
        edit.setText(shown)
        edit.blockSignals(False)


def save(app, win, name):
    win.show()
    app.processEvents()
    out = ROOT / "docs" / "images" / name
    out.parent.mkdir(exist_ok=True)
    win.grab().save(str(out))
    print(f"wrote {out}")
    win.close()


def main():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from hdeeg_snpm.gui.window import MainWindow

    runpy.run_path(str(ROOT / "examples" / "make_turtlewave_example.py"), run_name="__main__")
    make_table()
    app = QApplication.instance() or QApplication([])

    win = MainWindow()
    win.resize(1100, 760)
    win.load_folder(STUDY / "databases")
    win.load_subjects(STUDY / "subjects.csv")
    win.out_edit.setText(str(ROOT / "example_output"))
    win.seed.setText("20261025")
    win.permutations.setValue(1000)
    # Neutral names in the picture; the state behind them is the real one.
    neutral(((win.folder_edit, "/data/study/databases"),
             (win.subjects_edit, "/data/study/subjects.csv"), (win.out_edit, "/data/results")))
    save(app, win, "gui-window.png")

    win = MainWindow()
    win.resize(1100, 760)
    win.source.setCurrentIndex(win.source.findData("table"))
    win.analysis.setCurrentIndex(win.analysis.findData("regression"))
    win.load_table(1, TABLE)
    win.predictor_col.setCurrentIndex(win.predictor_col.findData("logAHI"))
    for i in range(win.covariates.count()):
        if win.covariates.item(i).text() in ("age", "sex"):
            win.covariates.item(i).setCheckState(Qt.Checked)
    win.out_edit.setText(str(ROOT / "example_output"))
    win.seed.setText("20261026")
    win.permutations.setValue(1000)
    neutral(((win.file_edit[1], "/data/study/band_power.csv"), (win.out_edit, "/data/results")))
    save(app, win, "gui-window-table.png")


if __name__ == "__main__":
    main()
