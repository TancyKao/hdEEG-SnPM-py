"""The PyQt5 window, offscreen: loading a synthetic study fills the choices, each analysis shows
its roles, Run waits until the form is complete, a run writes the four files, and a refusal from
the pipeline is shown with its text. Skipped when PyQt5 is not installed."""

import json
import os
import sqlite3

import numpy as np
import pytest

from hdeeg_snpm import cli
from test_turtlewave_config import SUBJECTS, same, study, subjects_csv

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtCore = pytest.importorskip("PyQt5.QtCore")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
MainWindow = pytest.importorskip("hdeeg_snpm.gui.window").MainWindow
QEventLoop, Qt, QTimer = QtCore.QEventLoop, QtCore.Qt, QtCore.QTimer

APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def shown(win, key):
    return not win.role_rows[key].isHidden()


def choose(combo, value):
    combo.setCurrentIndex(combo.findData(value))


def wait(win, timeout_ms=120_000):
    loop = QEventLoop()
    win.worker.finished.connect(loop.quit)
    QTimer.singleShot(timeout_ms, loop.quit)
    if win.worker.isRunning():
        loop.exec_()
    APP.processEvents()
    assert win.worker.isFinished()


@pytest.fixture
def unpaired(tmp_path):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", group=["A", "B"] * 6)
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    yield win, tmp_path
    win.close()


def test_loading_the_study_fills_the_event_choices(unpaired):
    win, _ = unpaired
    assert win.summary.text().startswith("12 databases, 12 subjects, sessions: (none).")
    assert win.recordings.rowCount() == 12 and win.recordings.item(0, 0).text() == "s01"
    assert win.event_type.currentData() == "spindle"
    assert win.method.currentData() == "Moelle2011" and win.band.currentText() == "12-15 Hz"
    assert [win.stages.item(i).text() for i in range(win.stages.count())] == ["NREM2", "NREM3"]
    assert all(win.stages.item(i).checkState() == Qt.Checked for i in range(2))


def test_run_waits_until_the_form_is_complete(unpaired):
    win, tmp = unpaired
    assert not win.run_btn.isEnabled()
    assert "a subjects file" in win.hint.text() and "an output folder" in win.hint.text()
    win.load_subjects(tmp / "subjects.csv")
    assert win.group_col.currentData() == "group"
    assert (win.group1.currentData(), win.group2.currentData()) == ("A", "B")
    win.out_edit.setText(str(tmp / "out"))
    assert win.run_btn.isEnabled() and win.hint.text() == "Ready to run."


def test_each_analysis_shows_its_roles(tmp_path):
    study(tmp_path / "db", sessions=("ses-1", "ses-2"))
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    expected = {"pairedT": {"sessions"},
                "unpairedT": {"subjects_file", "group_col", "groups", "session"},
                "anova1": {"subjects_file", "group_col", "session"},
                "correlationP": {"subjects_file", "behavior_col", "session"},
                "correlationS": {"subjects_file", "behavior_col", "session"}}
    for comparison, roles in expected.items():
        choose(win.analysis, comparison)
        assert {k for k in win.role_rows if shown(win, k)} == roles, comparison
        assert win.tail.isHidden() == (comparison not in ("pairedT", "unpairedT"))
    choose(win.analysis, "pairedT")
    assert (win.session1.currentData(), win.session2.currentData()) == ("ses-1", "ses-2")
    win.out_edit.setText(str(tmp_path / "out"))
    assert win.run_btn.isEnabled()
    win.close()


def test_a_run_writes_the_four_files_and_equals_the_saved_config(unpaired):
    win, tmp = unpaired
    win.load_subjects(tmp / "subjects.csv")
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(100)
    win.seed.setText("7")
    win.save_config(tmp / "saved.toml")
    win.start_run()
    assert not win.run_btn.isEnabled()
    wait(win)
    assert win.paths is not None, win.log.toPlainText()
    assert all(p.is_file() for p in win.paths.values()) and len(win.paths) == 4
    assert win.open_btn.isEnabled() and win.run_btn.isEnabled()
    text = (tmp / "saved.toml").read_text()
    out = json.dumps(str((tmp / "out").resolve()))
    assert out in text
    (tmp / "cli.toml").write_text(text.replace(out, json.dumps(str(tmp / "cli_out"))))
    assert cli.main(["run", str(tmp / "cli.toml")]) == 0
    same(np.load(win.paths["npz"]), np.load(next((tmp / "cli_out").glob("*.npz"))))


def test_a_refusal_is_shown_with_its_text(unpaired):
    win, tmp = unpaired
    subjects_csv(tmp / "short.csv", subjects=SUBJECTS[:-1], group=["A", "B"] * 5 + ["A"])
    win.load_subjects(tmp / "short.csv")
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(100)
    win.start_run()
    wait(win)
    assert win.paths is None
    assert "must name the same subjects" in win.error_box.text()
    assert "s12" in win.error_box.text() and "Stopped:" in win.log.toPlainText()
    win.error_box.close()


def test_a_path_typed_but_not_read_is_not_used(unpaired):
    win, tmp = unpaired
    study(tmp / "other", seed=5)
    win.load_subjects(tmp / "subjects.csv")
    win.out_edit.setText(str(tmp / "out"))
    assert win.run_btn.isEnabled()
    win.folder_edit.setText(str(tmp / "other"))          # typed, not yet read
    assert not win.run_btn.isEnabled() and not win.save_btn.isEnabled()
    assert "press Enter or Load" in win.hint.text() and win.state.to_params()["turtlewave"][
        "folder"] == ""
    win.load_folder(force=True)                          # the Load button
    assert win.state.folder == str(tmp / "other") and win.run_btn.isEnabled()
    subjects_csv(tmp / "cohort.csv", cohort=["X", "Y"] * 6)
    win.subjects_edit.setText(str(tmp / "cohort.csv"))
    assert not win.run_btn.isEnabled() and win.state.subjects_file == ""
    win.load_subjects()                                  # Enter in the field
    assert win.state.subjects_file == str(tmp / "cohort.csv")
    assert win.group_col.currentData() != "group"


def test_leaving_an_unchanged_field_keeps_the_choices(unpaired):
    win, _ = unpaired
    win.stages.item(0).setCheckState(Qt.Unchecked)
    choose(win.measure, "count")
    win.folder_edit.editingFinished.emit()
    assert win.stages.item(0).checkState() == Qt.Unchecked
    assert win.measure.currentData() == "count" and win.state.stage == ["NREM3"]


def test_no_stage_ticked_disables_run(unpaired):
    win, tmp = unpaired
    win.load_subjects(tmp / "subjects.csv")
    win.out_edit.setText(str(tmp / "out"))
    for i in range(win.stages.count()):
        win.stages.item(i).setCheckState(Qt.Unchecked)
    assert not win.run_btn.isEnabled() and "at least one stage" in win.hint.text()


def test_group_codes_with_leading_zeros_and_a_blank_run(unpaired):
    win, tmp = unpaired
    subjects_csv(tmp / "codes.csv", group=["01", "02"] * 5 + ["01", ""])
    win.load_subjects(tmp / "codes.csv")
    assert [win.group1.itemData(i) for i in range(1, win.group1.count())] == ["01", "02"]
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(20)
    win.start_run()
    wait(win)
    assert win.paths is not None, win.log.toPlainText()
    assert "s12 (empty)" in win.log.toPlainText()


def test_a_run_stored_without_a_band_shows_no_band(unpaired):
    win, tmp = unpaired
    with sqlite3.connect(tmp / "db" / "s01" / "wonambi" / "neural_events.db") as con:
        con.execute("INSERT INTO events VALUES ('E1', 'slow_wave', 'Massimini2004', 'NREM3', "
                    "NULL, NULL, 1.0, 50.0)")
    win.load_folder(force=True)
    choose(win.event_type, "slow_wave")
    assert win.band.count() == 0 and win.state.band is None
    assert "freq_lower" not in win.state.to_params()["turtlewave"]


def test_closing_during_a_run_waits_for_it_and_the_seed_is_drawn(unpaired):
    win, tmp = unpaired
    win.load_subjects(tmp / "subjects.csv")
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(2000)
    win.show()
    win.start_run()
    assert win.seed.text().isdigit() and win.state.seed == int(win.seed.text())
    first = win.worker
    win.start_run()                                      # ignored while running
    assert win.worker is first and not win.input_boxes[0].isEnabled()
    win.close()
    assert win.isVisible() and win.worker.isRunning()
    wait(win)
    assert not win.isVisible() and win.paths is not None
    assert all(p.is_file() for p in win.paths.values())


def test_a_run_that_fails_after_a_close_keeps_the_window_open(unpaired):
    win, tmp = unpaired
    subjects_csv(tmp / "short.csv", subjects=SUBJECTS[:-1], group=["A", "B"] * 5 + ["A"])
    win.load_subjects(tmp / "short.csv")
    win.out_edit.setText(str(tmp / "out"))
    win.show()
    win.start_run()
    win.close()
    wait(win)
    assert win.isVisible() and "must name the same subjects" in win.error_box.text()
    assert "The run failed, so the window stays open." in win.log.toPlainText()
    win.error_box.close()


def test_a_new_subject_id_source_keeps_the_stage_ticks(unpaired):
    win, _ = unpaired
    win.stages.item(0).setCheckState(Qt.Unchecked)
    choose(win.subject_from, "folder")
    assert win.stages.item(0).checkState() == Qt.Unchecked and win.state.stage == ["NREM3"]
