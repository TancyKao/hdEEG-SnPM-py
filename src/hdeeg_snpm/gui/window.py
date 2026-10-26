"""The PyQt5 window: a thin view on ``form.FormState``.

The window lists what the databases hold, collects the choices into a ``FormState`` and hands
``FormState.to_params()`` to ``output.run_and_write`` in a worker thread. It checks nothing
itself: every refusal comes from ``io.turtlewave_config.prepare`` and the pipeline, and is shown
with its own text. While a run is in progress the inputs are locked and the window does not
close until the run has finished, so no run is cut short and no output is left half written.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QRegExp, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QRegExpValidator
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..io import turtlewave as tw
from ..output import run_and_write
from . import form

DENSITY_TIP = ("Density is computed by TurtleWave itself; install it to use density "
               "(pip install \"hdeeg-snpm[turtlewave]\").")


UNSET = "\x00unset"     # data of a "choose ..." placeholder; "" is a real session (none)


def run_job(params: dict) -> tuple[dict, list[str]]:
    """Run the analysis and return its files and the notes to show (the notes and warnings the
    summary file records). The worker thread calls this, and so can a test. Warnings are not
    caught here: that would change a process-wide setting from a second thread."""
    paths = run_and_write(params)
    return paths, json.loads(Path(paths["json"]).read_text()).get("notes", [])


class RunWorker(QThread):
    done = pyqtSignal(object, object)
    failed = pyqtSignal(str)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self.params = params

    def run(self):
        try:
            paths, log = run_job(self.params)
        except ValueError as err:
            self.failed.emit(str(err))
        except Exception as err:     # noqa: BLE001 - shown, never a crash of the window
            self.failed.emit(f"{type(err).__name__}: {err}")
        else:
            self.done.emit(paths, log)


def _fill(combo, items, placeholder=None, select=None):
    """Replace a combo's items: ``items`` are (label, value) pairs or values."""
    combo.blockSignals(True)
    combo.clear()
    if placeholder:
        combo.addItem(placeholder, UNSET)
    for it in items:
        label, value = it if isinstance(it, tuple) and len(it) == 2 and \
            isinstance(it[0], str) else (str(it), it)
        combo.addItem(label, value)
    if select is not None:
        i = combo.findData(select)
        combo.setCurrentIndex(max(i, 0))
    combo.blockSignals(False)


def _data(combo):
    """The chosen value, or None for a placeholder or an empty combo."""
    value = combo.currentData()
    return None if value is None or value == UNSET else value


def _row(*widgets):
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    return w


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("hdeeg-snpm: analyse TurtleWave databases")
        self.state = form.FormState()
        self.available = self.subjects = None
        self.loaded_folder = self.loaded_subjects = ""     # the paths read, not the text shown
        self.worker = None
        self.paths = None
        self.close_when_done = False

        # 1. Study
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("folder searched for neural_events.db")
        self.folder_edit.editingFinished.connect(lambda: self.load_folder())
        self.folder_edit.textChanged.connect(self.refresh)
        load = QPushButton("Load")
        load.clicked.connect(lambda: self.load_folder(force=True))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_folder)
        self.subject_from = QComboBox()
        _fill(self.subject_from, [("auto", "auto"), ("database", "database"),
                                  ("folder name", "folder")])
        self.subject_from.currentIndexChanged.connect(lambda: self.load_folder(force=True))
        self.summary = QLabel("No study loaded.")
        self.summary.setWordWrap(True)
        self.recordings = QTableWidget(0, 3)
        self.recordings.setHorizontalHeaderLabels(["Subject", "Session", "Subject ID from"])
        self.recordings.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.recordings.horizontalHeader().setStretchLastSection(True)
        self.recordings.verticalHeader().setVisible(False)
        study = QFormLayout()
        study.addRow("Study folder", _row(self.folder_edit, load, browse))
        study.addRow("Subject ID from", self.subject_from)
        study.addRow(self.summary)
        study.addRow(self.recordings)

        # 2. Events
        self.event_type = QComboBox()
        self.event_type.currentIndexChanged.connect(self._fill_events)
        self.method = QComboBox()
        self.stages = QListWidget()
        self.stages.setMaximumHeight(90)
        self.stages.itemChanged.connect(self.refresh)
        self.band = QComboBox()
        self.measure = QComboBox()
        _fill(self.measure, tw.MEASURES, select="amplitude")
        if not form.density_available():
            item = self.measure.model().item(self.measure.findData("density"))
            item.setEnabled(False)
            item.setToolTip(DENSITY_TIP)
        events = QFormLayout()
        events.addRow("Event type", self.event_type)
        events.addRow("Method", self.method)
        events.addRow("Stages", self.stages)
        events.addRow("Band", self.band)
        events.addRow("Measure", self.measure)

        # 3. Analysis
        self.analysis = QComboBox()
        _fill(self.analysis, [(v, k) for k, v in form.ANALYSES.items()], select="unpairedT")
        self.tail = QComboBox()
        _fill(self.tail, [("two-sided", "both"), ("left", "left"), ("right", "right")])
        self.permutations = QSpinBox()
        self.permutations.setRange(20, 1_000_000)
        self.permutations.setValue(5000)
        self.seed = QLineEdit()
        self.seed.setPlaceholderText("drawn and recorded when empty")
        self.seed.setValidator(QRegExpValidator(QRegExp(r"\d{0,19}")))
        self.system = QComboBox()
        _fill(self.system, [("EGI 256 HydroCel", "egi"), ("Compumedics 257 Neuvo", "compu")])
        self.datatype = QComboBox()
        _fill(self.datatype, [("absolute (no transform)", "absolute"), ("log10", "logscale"),
                              ("z-score within subject", "normalize")])
        analysis = QFormLayout()
        analysis.addRow("Analysis", self.analysis)
        analysis.addRow("Tail", self.tail)
        analysis.addRow("Permutations", self.permutations)
        analysis.addRow("Seed", self.seed)
        analysis.addRow("Recording system", self.system)
        analysis.addRow("Data type", self.datatype)
        self.analysis_form = analysis

        # 4. Roles
        self.session1, self.session2 = QComboBox(), QComboBox()
        self.subjects_edit = QLineEdit()
        self.subjects_edit.setPlaceholderText("Subject column plus groups or scores")
        self.subjects_edit.editingFinished.connect(lambda: self.load_subjects())
        self.subjects_edit.textChanged.connect(self.refresh)
        load_subjects = QPushButton("Load")
        load_subjects.clicked.connect(lambda: self.load_subjects(force=True))
        browse_subjects = QPushButton("Browse…")
        browse_subjects.clicked.connect(self._browse_subjects)
        self.group_col = QComboBox()
        self.group_col.currentIndexChanged.connect(self._fill_groups)
        self.group1, self.group2 = QComboBox(), QComboBox()
        self.score_col = QComboBox()
        self.session = QComboBox()
        self.roles = QFormLayout()
        self.role_rows = {
            "sessions": _row(QLabel("Data 1"), self.session1, QLabel("Data 2"), self.session2),
            "subjects_file": _row(self.subjects_edit, load_subjects, browse_subjects),
            "group_col": self.group_col,
            "groups": _row(QLabel("Data 1"), self.group1, QLabel("Data 2"), self.group2),
            "behavior_col": self.score_col,
            "session": self.session,
        }
        for key, label in (("sessions", "Sessions"), ("subjects_file", "Subjects file"),
                           ("group_col", "Group column"), ("groups", "Groups"),
                           ("behavior_col", "Score column"),
                           ("session", "Session to analyse")):
            self.roles.addRow(label, self.role_rows[key])

        # 5. Output and run
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("folder for the results")
        browse_out = QPushButton("Browse…")
        browse_out.clicked.connect(self._browse_out)
        self.save_btn = QPushButton("Save config…")
        self.save_btn.clicked.connect(self._save_dialog)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.start_run)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        output = QFormLayout()
        output.addRow("Output folder", _row(self.out_edit, browse_out))
        output.addRow(_row(self.save_btn, self.run_btn))
        output.addRow(self.hint)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.files = QLabel()
        self.files.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.files.setWordWrap(True)
        self.open_btn = QPushButton("Open report")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_report)
        log = QVBoxLayout()
        log.addWidget(self.log)
        log.addWidget(self.files)
        log.addWidget(self.open_btn)

        for w in (self.method, self.band, self.measure, self.analysis, self.tail, self.system,
                  self.datatype, self.session1, self.session2, self.group1, self.group2,
                  self.score_col, self.session):
            w.currentIndexChanged.connect(self.refresh)
        self.permutations.valueChanged.connect(self.refresh)
        for w in (self.seed, self.out_edit):
            w.textChanged.connect(self.refresh)

        left, right = QVBoxLayout(), QVBoxLayout()
        self.input_boxes = []
        for lay, (title, inner) in [(left, ("1. Study", study)), (left, ("2. Events", events)),
                                    (right, ("3. Analysis", analysis)),
                                    (right, ("4. Roles", self.roles)),
                                    (right, ("5. Output", output)), (right, ("Log", log))]:
            box = QGroupBox(title)
            box.setLayout(inner)
            lay.addWidget(box)
            if title != "Log":
                self.input_boxes.append(box)
        top = QHBoxLayout(self)
        top.addLayout(left, 1)
        top.addLayout(right, 1)
        self.refresh()

    # Loading what the databases and the subjects file hold
    def running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def load_folder(self, path=None, force=False):
        """Read the folder in the field. Leaving the field without changing it keeps what was
        read and chosen; ``force`` (the Load button, a new subject-ID source) reads again."""
        if self.running():
            return
        if path is not None:
            self.folder_edit.setText(str(path))
        text = self.folder_edit.text().strip()
        if text == self.loaded_folder and not force:
            return
        self.loaded_folder, self.state.sessions_found = "", []
        self.recordings.setRowCount(0)
        if not text:
            self.summary.setText("No study loaded.")
            self.refresh()
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                recs = tw.find_recordings(text, self.subject_from.currentData())
                self.available = tw.available(recs)
        except Exception as err:     # noqa: BLE001 - shown; the folder stays unloaded
            self.summary.setText(f"Could not read the study: {err}")
            self.refresh()
            return
        finally:
            QApplication.restoreOverrideCursor()
        sessions = sorted({r.session for r in recs})
        self.loaded_folder, self.state.sessions_found = text, sessions
        self.summary.setText(
            f"{len(recs)} databases, {len({r.subject for r in recs})} subjects, sessions: "
            f"{', '.join(s or '(none)' for s in sessions)}."
            + "".join(f"\n{w.message}" for w in caught))
        self.recordings.setRowCount(len(recs))
        for i, r in enumerate(recs):
            for j, v in enumerate((r.subject, r.session, r.subject_source)):
                self.recordings.setItem(i, j, QTableWidgetItem(v))
        labelled = [(s or "(none)", s) for s in sessions]
        _fill(self.session1, labelled, select=sessions[0])
        _fill(self.session2, labelled, select=sessions[min(1, len(sessions) - 1)])
        _fill(self.session, labelled, placeholder="choose a session")
        _fill(self.event_type, sorted(self.available["event_type"].unique()))
        self._fill_events()

    def _fill_events(self):
        if not self.loaded_folder:
            return
        c = form.choices(self.available, _data(self.event_type) or "")
        ticked = {self.stages.item(i).text(): self.stages.item(i).checkState()
                  for i in range(self.stages.count())}
        if set(ticked) != set(c["stages"]):
            ticked = {}      # other stages now: tick them all
        _fill(self.method, c["methods"])
        _fill(self.band, [(f"{lo:g}-{hi:g} Hz", (lo, hi)) for lo, hi in c["bands"]])
        self.stages.blockSignals(True)
        self.stages.clear()
        for s in c["stages"]:
            item = QListWidgetItem(s)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(ticked.get(s, Qt.Checked))
            self.stages.addItem(item)
        self.stages.blockSignals(False)
        self.state.stages_found = c["stages"]
        self.refresh()

    def load_subjects(self, path=None, force=False):
        """Read the subjects file in the field; as ``load_folder``, an unchanged field keeps
        what was read and chosen."""
        if self.running():
            return
        if path is not None:
            self.subjects_edit.setText(str(path))
        text = self.subjects_edit.text().strip()
        if text == self.loaded_subjects and not force:
            return
        self.subjects, self.loaded_subjects = None, ""
        try:
            self.subjects = form.subjects_table(text) if text else None
            self.loaded_subjects = text
        except Exception as err:     # noqa: BLE001 - shown; the file stays unloaded
            self.log.appendPlainText(f"Could not read the subjects file: {err}")
        cols = form.info_columns(self.subjects) if self.subjects is not None else []
        _fill(self.group_col, cols, placeholder="choose a column",
              select="group" if "group" in cols else None)
        _fill(self.score_col, cols, placeholder="choose a column",
              select=cols[0] if len(cols) == 1 else None)
        self._fill_groups()

    def _fill_groups(self):
        col = _data(self.group_col)
        values = (sorted(self.subjects[col].dropna().unique())
                  if col and self.subjects is not None else [])
        _fill(self.group1, values, placeholder="choose", select=values[0] if values else None)
        _fill(self.group2, values, placeholder="choose",
              select=values[1] if len(values) > 1 else None)
        self.refresh()

    # The form state and what is shown
    def _read(self):
        s = self.state
        s.subject_from = self.subject_from.currentData()
        # A path typed but not read yet counts as no path, so the run never uses a folder or a
        # subjects file other than the one shown.
        same = self.folder_edit.text().strip() == self.loaded_folder
        s.folder = self.loaded_folder if same else ""
        same = self.subjects_edit.text().strip() == self.loaded_subjects
        s.subjects_file = self.loaded_subjects if same and self.subjects is not None else ""
        s.event_type = _data(self.event_type) or ""
        s.method = _data(self.method) or ""
        band = _data(self.band)
        s.band = tuple(band) if band else None
        s.stage = [self.stages.item(i).text() for i in range(self.stages.count())
                   if self.stages.item(i).checkState() == Qt.Checked]
        s.measure = _data(self.measure) or ""
        s.comparison = _data(self.analysis)
        s.tail = _data(self.tail)
        s.permutations = self.permutations.value()
        s.seed = int(self.seed.text()) if self.seed.text() else None
        s.channels, s.datatype = _data(self.system), _data(self.datatype)
        s.sessions = [_data(self.session1), _data(self.session2)]
        s.session = _data(self.session)
        s.group_col = _data(self.group_col) or ""
        s.groups = [_data(self.group1) or "", _data(self.group2) or ""]
        s.behavior_col = _data(self.score_col) or ""
        s.output_path = self.out_edit.text().strip()

    def refresh(self, *_):
        self._read()
        shown = self.state.roles()
        for key, w in self.role_rows.items():
            w.setVisible(key in shown)
            self.roles.labelForField(w).setVisible(key in shown)
        is_t = self.state.comparison in form.T_TESTS
        self.tail.setVisible(is_t)
        self.analysis_form.labelForField(self.tail).setVisible(is_t)
        missing, running = self.state.missing(), self.running()
        for box in self.input_boxes:
            box.setEnabled(not running)
        self.run_btn.setEnabled(not missing and not running)
        self.save_btn.setEnabled(not missing and not running)
        self.hint.setText("Running…" if running else
                          "To run, choose " + ", ".join(missing) + "." if missing
                          else "Ready to run.")

    # Actions
    def _ensure_seed(self):
        """Draw the seed here when none is given, and show it, so a saved config repeats the
        run exactly (the same rule ``run_and_write`` uses)."""
        if not self.seed.text():
            self.seed.setText(str(int(np.random.SeedSequence().generate_state(1)[0])))
        self.refresh()

    def save_config(self, path):
        self._ensure_seed()
        Path(path).write_text(self.state.to_toml(), encoding="utf-8")
        self.log.appendPlainText(f"Saved the config to {path}. Repeat with: snpm run {path}")

    def start_run(self):
        if self.running():
            return
        self.refresh()
        if self.state.missing():
            return
        self._ensure_seed()
        self.paths = None
        self.files.setText("")
        self.open_btn.setEnabled(False)
        self.log.appendPlainText("Running…")
        self.worker = RunWorker(self.state.to_params(), self)
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._run_ended)
        self.worker.start()
        self.refresh()

    def _run_ended(self):
        self.refresh()
        if self.close_when_done and self.paths is None:
            # The run failed: keep the window, so its error stays on screen.
            self.close_when_done = False
            self.log.appendPlainText("The run failed, so the window stays open.")
        elif self.close_when_done:
            self.close()

    def closeEvent(self, event):
        """A running analysis is never cut short: the window stays open until it finishes and
        then closes by itself."""
        if self.running():
            self.close_when_done = True
            self.log.appendPlainText("A run is in progress; the window closes when it finishes.")
            event.ignore()
            return
        super().closeEvent(event)

    def _finished(self, paths, log):
        self.paths = paths
        for line in log:
            self.log.appendPlainText(line)
        self.log.appendPlainText("Done.")
        self.files.setText("\n".join(str(p) for p in paths.values()))
        self.open_btn.setEnabled(True)

    def _failed(self, message):
        self.log.appendPlainText(f"Stopped: {message}")
        self.error_box = QMessageBox(QMessageBox.Critical, "The analysis did not run", message,
                                     parent=self)
        self.error_box.open()

    def open_report(self):
        if self.paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.paths["html"])))

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Study folder")
        if path:
            self.load_folder(path, force=True)

    def _browse_subjects(self):
        path, _ = QFileDialog.getOpenFileName(self, "Subjects file", "",
                                              "Tables (*.csv *.xlsx *.xls *.xlsm)")
        if path:
            self.load_subjects(path, force=True)

    def _browse_out(self):
        path = QFileDialog.getExistingDirectory(self, "Output folder")
        if path:
            self.out_edit.setText(path)

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save config", "snpm_run.toml",
                                              "TOML (*.toml)")
        if path:
            self.save_config(path)


def main(argv=None) -> int:
    app = QApplication.instance() or QApplication(["snpm-gui", *(argv or [])])
    win = MainWindow()
    win.resize(1100, 760)
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
