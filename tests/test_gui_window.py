"""The PyQt5 window, offscreen: loading a synthetic study fills the choices, each analysis shows
its roles, Run waits until the form is complete, a run writes the four files, and a refusal from
the pipeline is shown with its text. Skipped when PyQt5 is not installed."""

import json
import os
import sqlite3

import numpy as np
import pytest

from hdeeg_snpm import cli
from test_gui_form import DESIGN_META, EXAMPLES, table_study
from test_turtlewave import make_db
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
                "ancova": {"subjects_file", "group_col", "covariate_cols", "session"},
                "regression": {"subjects_file", "predictor_col", "covariate_cols", "session"},
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


def test_a_subject_id_disagreement_is_shown_in_window_words(tmp_path):
    study(tmp_path / "db")
    with sqlite3.connect(tmp_path / "db" / "s01" / "wonambi" / "neural_events.db") as con:
        con.execute("CREATE TABLE stage_durations (subject TEXT)")
        con.execute("INSERT INTO stage_durations VALUES ('sub-99zz')")
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    text = win.summary.text()
    assert text.startswith("Could not read the study:") and "sub-99zz" in text
    assert "set Subject ID from to folder or database" in text and "subject_from" not in text
    choose(win.subject_from, "folder")
    assert win.summary.text().startswith("12 databases")
    win.close()


def test_the_reject_set_is_shown_for_density(tmp_path):
    for s in ("s01", "s02"):
        make_db(tmp_path / "db" / s / "wonambi", [("E1", 1.0, 40.0)], s,
                analysed={(1, 0): {"NREM2": 600.0}})
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    assert win.reject.text() == "Artefact (as detected)"
    assert win.reject_row.isHidden()                     # amplitude: not shown
    choose(win.measure, "density")       # set even when the item is disabled (no TurtleWave)
    assert not win.reject_row.isHidden() and not win.reject.isHidden()
    assert win.reject_choice.isHidden() and "reject_types" not in win.state.to_params()[
        "turtlewave"]
    win.close()


def test_a_reject_set_must_be_chosen_when_databases_store_several(tmp_path):
    for s in ("s01", "s02"):
        make_db(tmp_path / "db" / s / "wonambi", [("E1", 1.0, 40.0)], s,
                analysed={(1, 0): {"NREM2": 600.0}, (1, 1): {"NREM2": 500.0}})
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    win.out_edit.setText(str(tmp_path / "out"))
    choose(win.measure, "density")
    assert win.reject.isHidden() and not win.reject_choice.isHidden()
    assert [win.reject_choice.itemData(i) for i in range(1, win.reject_choice.count())] == \
        ["Arousal,Artefact", "Artefact"]
    assert win.reject_choice.currentData() == "\x00unset"          # no default
    assert not win.run_btn.isEnabled() and "the reject set of density" in win.hint.text()
    choose(win.reject_choice, "Artefact")
    assert win.state.to_params()["turtlewave"]["reject_types"] == "Artefact"
    win.close()


def test_run_is_blocked_when_no_reject_set_is_shared(tmp_path):
    make_db(tmp_path / "db" / "s01" / "wonambi", [("E1", 1.0, 40.0)], "s01",
            analysed={(1, 0): {"NREM2": 600.0}, (1, 1): {"NREM2": 500.0}})
    make_db(tmp_path / "db" / "s02" / "wonambi", [("E1", 1.0, 40.0)], "s02",
            analysed={(0, 0): {"NREM2": 600.0}, (0, 1): {"NREM2": 500.0}})
    win = MainWindow()
    win.load_folder(tmp_path / "db")
    win.out_edit.setText(str(tmp_path / "out"))
    choose(win.measure, "density")
    assert win.reject_choice.isHidden() and win.reject.text().startswith(
        "no reject set is stored by every database")
    assert not win.run_btn.isEnabled() and "share a reject set" in win.hint.text()
    choose(win.measure, "count")
    assert "share a reject set" not in win.hint.text()
    win.close()


def ticked(win):
    return [win.covariates.item(i).text() for i in range(win.covariates.count())
            if win.covariates.item(i).checkState() == Qt.Checked]


def tick(win, *names):
    for i in range(win.covariates.count()):
        if win.covariates.item(i).text() in names:
            win.covariates.item(i).setCheckState(Qt.Checked)


@pytest.fixture
def table(tmp_path):
    table_study(tmp_path)
    win = MainWindow()
    choose(win.source, "table")
    yield win, tmp_path
    win.close()


def test_table_source_shows_the_files_and_their_detection(table):
    win, tmp = table
    assert win.boxes["Study"].isHidden() and win.boxes["Events"].isHidden()
    assert not win.boxes["Table files"].isHidden()
    assert win.boxes["Table files"].title() == "1. Table files"
    assert win.boxes["Analysis"].title() == "2. Analysis"
    analyses = [win.analysis.itemData(i) for i in range(win.analysis.count())]
    assert "onesampleT" in analyses and "regression" in analyses and "ancova" in analyses
    choose(win.analysis, "regression")
    assert win.file_rows[2].isHidden() and not win.file_rows[1].isHidden()
    assert win.table_form.labelForField(win.file_rows[1]).text() == "Data file"
    assert win.sheet[1].isHidden()                                  # a CSV has no sheets
    win.load_table(1, tmp / "design.csv")
    assert win.caption[1].text() == (
        "Detected: 256 channels, 5 other columns: Subject, group, logAHI, age, sex. "
        "177 of them are EGI 256 (HydroCel) channels.")
    choose(win.system, "compu")
    assert win.caption[1].text().endswith("No data columns match the Compumedics 257 (Neuvo) "
                                          "montage labels.")
    cols = DESIGN_META[1:]
    assert [win.predictor_col.itemData(i) for i in range(1, win.predictor_col.count())] == \
        ["logAHI", "age"]                                   # numeric columns only
    assert [win.covariates.item(i).text() for i in range(win.covariates.count())] == cols
    assert win.group_col.currentData() == "group" and win.state.meta_cols == DESIGN_META
    choose(win.analysis, "correlationP")
    assert not win.file_rows[2].isHidden()
    assert win.table_form.labelForField(win.file_rows[2]).text() == "Score file"
    assert win.boxes["Roles"].isHidden() and win.boxes["Output"].title() == "3. Output"
    choose(win.source, "turtlewave")
    assert win.analysis.currentData() == "correlationP" and win.boxes["Table files"].isHidden()
    assert win.boxes["Study"].title() == "1. Study" and win.boxes["Output"].title() == "5. Output"


def test_table_roles_per_analysis_and_run_waits_for_them(table):
    win, tmp = table
    expected = {"pairedT": set(), "unpairedT": set(), "onesampleT": set(),
                "anova1": {"group_col"}, "ancova": {"group_col", "covariate_cols"},
                "regression": {"predictor_col", "covariate_cols"},
                "correlationP": set(), "correlationS": set()}
    for comparison, roles in expected.items():
        choose(win.analysis, comparison)
        assert {k for k in win.role_rows if shown(win, k)} == roles, comparison
    win.load_table(1, tmp / "design.csv")
    win.out_edit.setText(str(tmp / "out"))
    choose(win.analysis, "regression")
    assert not win.run_btn.isEnabled() and "a predictor column" in win.hint.text()
    choose(win.predictor_col, "logAHI")
    assert win.run_btn.isEnabled()
    choose(win.analysis, "ancova")
    assert not win.run_btn.isEnabled() and "at least one covariate column" in win.hint.text()
    tick(win, "age", "sex")
    assert win.run_btn.isEnabled()
    assert win.state.to_params()["covariate_cols"] == ["age", "sex"]
    choose(win.analysis, "unpairedT")
    assert "a group 2 file" in win.hint.text() and not win.run_btn.isEnabled()
    win.load_table(2, tmp / "condB.csv")
    assert win.run_btn.isEnabled()


def test_an_excel_sheet_is_chosen_and_read(table):
    win, tmp = table
    choose(win.analysis, "regression")
    win.load_table(1, tmp / "design.xlsx")
    assert not win.sheet[1].isHidden()
    assert [win.sheet[1].itemData(i) for i in range(win.sheet[1].count())] == ["notes", "data"]
    assert win.caption[1].text() == "Detected: 0 channels, 1 other column: note."
    choose(win.sheet[1], "data")
    assert win.caption[1].text().startswith("Detected: 256 channels, 5 other columns")
    assert win.state.data1_sheet == "data" and win.state.meta_cols == DESIGN_META
    choose(win.predictor_col, "logAHI")
    tick(win, "age")
    win.file_edit[1].editingFinished.emit()               # unchanged field: nothing re-read
    assert win.sheet[1].currentData() == "data" and ticked(win) == ["age"]
    win.out_edit.setText(str(tmp / "out"))
    params = win.state.to_params()
    assert params["data_file"] == str(tmp / "design.xlsx") and params["data_sheet"] == "data"
    win.load_table(1, tmp / "design.csv")                 # a CSV: no sheet any more
    assert win.sheet[1].isHidden() and "data_sheet" not in win.state.to_params()
    assert win.predictor_col.currentData() == "logAHI" and ticked(win) == ["age"]


def test_a_table_typed_but_not_read_is_not_used(table):
    win, tmp = table
    choose(win.analysis, "regression")
    win.load_table(1, tmp / "design.csv")
    choose(win.predictor_col, "logAHI")
    win.out_edit.setText(str(tmp / "out"))
    assert win.run_btn.isEnabled()
    win.file_edit[1].setText(str(tmp / "design.xlsx"))     # typed, not yet read
    assert not win.run_btn.isEnabled() and not win.save_btn.isEnabled()
    assert "a data file (press Enter or Load" in win.hint.text()
    assert "data_file" not in win.state.to_params()
    win.load_table(1)                                       # Enter in the field
    assert win.state.data1_file == str(tmp / "design.xlsx")
    win.file_edit[1].setText(str(tmp / "nothing.csv"))
    win.load_table(1)
    assert win.caption[1].text().startswith("Could not read the file:")
    assert not win.run_btn.isEnabled()


def test_a_table_regression_from_the_window_equals_the_saved_config(table):
    win, tmp = table
    choose(win.analysis, "regression")
    win.load_table(1, tmp / "design.csv")
    choose(win.predictor_col, "logAHI")
    tick(win, "age", "sex")
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(100)
    win.seed.setText("7")
    win.save_config(tmp / "saved.toml")
    win.start_run()
    wait(win)
    assert win.paths is not None, win.log.toPlainText()
    assert all(p.is_file() for p in win.paths.values()) and len(win.paths) == 4
    text = (tmp / "saved.toml").read_text()
    assert "[turtlewave]" not in text and 'covariate_cols = ["age", "sex"]' in text
    out = json.dumps(str((tmp / "out").resolve()))
    (tmp / "cli.toml").write_text(text.replace(out, json.dumps(str(tmp / "cli_out"))))
    assert cli.main(["run", str(tmp / "cli.toml")]) == 0
    same(np.load(win.paths["npz"]), np.load(next((tmp / "cli_out").glob("*.npz"))))


def test_a_table_refusal_comes_from_the_pipeline(table):
    win, tmp = table
    choose(win.analysis, "pairedT")
    win.load_table(1, tmp / "design.csv")                   # carries non-channel columns
    win.load_table(2, tmp / "condB.csv")
    win.out_edit.setText(str(tmp / "out"))
    win.permutations.setValue(100)
    assert win.run_btn.isEnabled()
    win.start_run()
    wait(win)
    assert win.paths is None and "Stopped:" in win.log.toPlainText()
    assert "do not list the same channels" in win.error_box.text()
    win.error_box.close()


def test_turtlewave_regression_roles_come_from_the_subjects_file(unpaired):
    win, tmp = unpaired
    subjects_csv(tmp / "subjects.csv", group=["A", "B"] * 6, score=np.linspace(0, 1, 12),
                 age=np.arange(20, 32) % 7)
    win.load_subjects(tmp / "subjects.csv")
    choose(win.analysis, "regression")
    assert [win.predictor_col.itemData(i) for i in range(1, win.predictor_col.count())] == \
        ["score", "age"]                                    # not the text column group
    win.out_edit.setText(str(tmp / "out"))
    assert "a predictor column" in win.hint.text()
    choose(win.predictor_col, "score")
    tick(win, "age")
    assert win.run_btn.isEnabled()
    params = win.state.to_params()
    assert params["predictor_col"] == "score" and params["covariate_cols"] == ["age"]
    assert "turtlewave" in params and "meta_cols" not in params


def test_the_predictor_menu_offers_only_numeric_columns(tmp_path):
    """On the shipped regression example the predictor menu offers logAHI and age, not the text
    columns group and sex, and starts on its placeholder rather than on a text column."""
    win = MainWindow()
    choose(win.source, "table")
    choose(win.analysis, "regression")
    win.load_table(1, EXAMPLES / "glm_regression.csv")
    assert win.predictor_col.currentIndex() == 0 and "a predictor column" in win.hint.text()
    assert [win.predictor_col.itemData(i) for i in range(1, win.predictor_col.count())] == \
        ["logAHI", "age"]
    choose(win.predictor_col, "logAHI")
    assert [win.covariates.item(i).text() for i in range(win.covariates.count())] == \
        ["group", "age", "sex"]
    win.close()


def test_the_effect_of_interest_is_not_offered_as_a_covariate(table):
    win, tmp = table
    choose(win.analysis, "regression")
    win.load_table(1, tmp / "design.csv")
    tick(win, "age", "sex")
    choose(win.predictor_col, "age")
    assert [win.covariates.item(i).text() for i in range(win.covariates.count())] == \
        ["group", "logAHI", "sex"]
    assert ticked(win) == ["sex"] and win.state.covariate_cols == ["sex"]
    choose(win.predictor_col, "logAHI")                      # age comes back, not ticked
    assert ticked(win) == ["sex"] and "logAHI" not in \
        [win.covariates.item(i).text() for i in range(win.covariates.count())]
    choose(win.analysis, "ancova")
    assert win.group_col.currentData() == "group"
    assert "group" not in [win.covariates.item(i).text() for i in range(win.covariates.count())]
    assert "logAHI" in [win.covariates.item(i).text() for i in range(win.covariates.count())]


def test_the_sheet_menu_belongs_to_the_file_read(table):
    win, tmp = table
    choose(win.analysis, "regression")
    win.load_table(1, tmp / "design.xlsx")
    choose(win.sheet[1], "data")
    win.file_edit[1].setText(str(tmp / "design.csv"))       # typed, not read
    assert win.sheet[1].isHidden()
    win.sheet[1].setCurrentIndex(win.sheet[1].findData("notes"))
    assert win.loaded_table[1] == str(tmp / "design.xlsx") and win.columns[1] != ["note"]
    assert win.file_edit[1].text() == str(tmp / "design.csv")
    win.file_edit[1].setText(str(tmp / "design.xlsx"))      # back to the file read
    assert not win.sheet[1].isHidden()


def test_a_second_file_read_for_another_analysis_is_not_sent(table):
    win, tmp = table
    choose(win.analysis, "pairedT")
    win.load_table(1, tmp / "condA.csv")
    win.load_table(2, tmp / "condB.csv")
    win.out_edit.setText(str(tmp / "out"))
    assert win.run_btn.isEnabled() and win.state.data2_file == str(tmp / "condB.csv")
    choose(win.analysis, "regression")                      # one file: the second is kept
    choose(win.analysis, "pairedT")
    assert win.state.data2_file == str(tmp / "condB.csv")
    choose(win.analysis, "correlationP")                    # now a score file: cleared
    assert win.file_edit[2].text() == "" and win.state.data2_file == ""
    assert "data2_file" not in win.state.to_params() and not win.run_btn.isEnabled()
    assert "a score file" in win.hint.text()
    win.load_table(2, tmp / "scores.csv")
    choose(win.analysis, "correlationS")                    # same meaning: kept
    assert win.state.to_params()["data2_file"] == str(tmp / "scores.csv")
