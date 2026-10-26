"""The window's form without Qt: the parameters it builds equal a hand-written `[turtlewave]`
config, its saved config reads back the same, and a run from it equals `snpm run`."""

import sqlite3
import subprocess
import sys
import warnings

import numpy as np
import pytest

from hdeeg_snpm import cli, gui
from hdeeg_snpm.gui import form
from hdeeg_snpm.io import turtlewave as tw
from hdeeg_snpm.output import run_and_write
from test_turtlewave import make_db
from test_turtlewave_config import SUBJECTS, run, same, study, subjects_csv

TOP = {
    "pairedT": 'comparison = "pairedT"\ntail = "both"\nsessions = ["ses-1", "ses-2"]',
    "unpairedT": 'comparison = "unpairedT"\ntail = "both"\ngroup_col = "group"\n'
                 'groups = ["A", "B"]',
    "anova1": 'comparison = "anova1"\ngroup_col = "group"',
    "correlationP": 'comparison = "correlationP"\nbehavior_col = "score"',
    "correlationS": 'comparison = "correlationS"\nbehavior_col = "score"',
}
ROLES = {
    "pairedT": {"sessions": ["ses-1", "ses-2"]},
    "unpairedT": {"group_col": "group", "groups": ["A", "B"]},
    "anova1": {"group_col": "group"},
    "correlationP": {"behavior_col": "score"},
    "correlationS": {"behavior_col": "score"},
}


def setup(tmp_path, comparison, out="out"):
    """The synthetic study, the form filled for ``comparison`` and the hand-written config."""
    paired = comparison == "pairedT"
    study(tmp_path / "db", sessions=("ses-1", "ses-2") if paired else ("",))
    groups = ["A", "B", "C"] * 4 if comparison == "anova1" else ["A", "B"] * 6
    subjects_csv(tmp_path / "subjects.csv", group=groups, score=np.linspace(0, 1, 12))
    state = form.FormState(
        folder=str(tmp_path / "db"), sessions_found=["ses-1", "ses-2"] if paired else [""],
        event_type="spindle", method="Moelle2011", stage=["NREM2", "NREM3"], band=(12.0, 15.0),
        measure="count", comparison=comparison, permutations=100, seed=7,
        subjects_file="" if paired else str(tmp_path / "subjects.csv"),
        output_path=str(tmp_path / out), **ROLES[comparison])
    text = TOP[comparison] + '\nchannels = "egi"\ndatatype = "absolute"\npermutations = 100\n' \
        f'seed = 7\noutput_path = "{out}"\n\n[turtlewave]\nfolder = "db"\n' \
        + ("" if paired else 'subjects_file = "subjects.csv"\n') + \
        'event_type = "spindle"\nmethod = "Moelle2011"\nstage = ["NREM2", "NREM3"]\n' \
        'freq_lower = 12\nfreq_upper = 15\nmeasure = "count"\nsubject_from = "auto"\n'
    (tmp_path / "hand.toml").write_text(text)
    return state, tmp_path / "hand.toml"


@pytest.mark.parametrize("comparison", list(form.ANALYSES))
def test_params_equal_the_hand_written_config_and_read_back(tmp_path, comparison):
    state, hand = setup(tmp_path, comparison)
    assert state.missing() == []
    assert state.to_params() == cli.load_config(hand)
    saved = tmp_path / "saved.toml"
    saved.write_text(state.to_toml())
    assert cli.load_config(saved) == state.to_params()


@pytest.mark.parametrize("comparison", list(form.ANALYSES))
def test_a_run_from_the_form_equals_snpm_run(tmp_path, comparison):
    state, hand = setup(tmp_path, comparison, out="gui_out")
    hand.write_text(hand.read_text().replace('"gui_out"', '"out"'))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = run_and_write(state.to_params())
    same(np.load(paths["npz"]), run(hand)[0])


def test_missing_lists_what_each_analysis_still_needs():
    folder, subjects = ("a study folder with databases (press Enter or Load to read it)",
                        "a subjects file (press Enter or Load to read it)")
    s = form.FormState()
    assert s.missing() == [folder, "an event type", subjects, "a group column",
                           "two different groups", "an output folder"]
    s = form.FormState(folder="db", event_type="spindle", output_path="out",
                       comparison="pairedT", sessions=["ses-1", "ses-1"])
    assert s.missing() == ["two different sessions"]
    s.comparison = "anova1"
    assert s.missing() == [subjects, "a group column"]
    s.comparison, s.subjects_file = "correlationS", "subjects.csv"
    assert s.missing() == ["a score column"]
    s.behavior_col, s.sessions_found = "score", ["ses-1", "ses-2"]
    assert s.roles() == ["subjects_file", "behavior_col", "session"]
    assert s.missing() == ["the session to analyse"]
    s.session = "ses-2"
    assert s.missing() == [] and s.to_params()["sessions"] == ["ses-2"]


def test_no_stage_ticked_is_missing_not_every_stage():
    s = form.FormState(folder="db", event_type="spindle", output_path="out",
                       comparison="pairedT", sessions=["ses-1", "ses-2"],
                       stages_found=["NREM2", "NREM3"])
    assert s.missing() == ["at least one stage"]
    s.stage = ["NREM3"]
    assert s.missing() == [] and s.to_params()["turtlewave"]["stage"] == ["NREM3"]


def test_recordings_without_a_session_folder_can_be_the_session_analysed():
    s = form.FormState(folder="db", event_type="spindle", output_path="out",
                       comparison="correlationP", subjects_file="s.csv", behavior_col="score",
                       sessions_found=["", "ses-1"])
    assert s.missing() == ["the session to analyse"]
    s.session = ""
    assert s.missing() == [] and s.to_params()["sessions"] == [""]


@pytest.mark.parametrize("name", ["a b", 'q"uote', "back\\slash", "\u00e9\u4e2d",
                                  "emoji\U0001F600", "tab\tx", "nl\nx", "del\x7fx"])
def test_saved_config_reads_back_with_any_folder_name(tmp_path, name):
    folder = tmp_path / name
    folder.mkdir()
    s = form.FormState(folder=str(folder), event_type="spindle", output_path=str(folder / "out"),
                       comparison="pairedT", sessions=["ses-1", "ses-2"], stage=["NREM2"],
                       seed=9999999999999999999, band=(12.0, 15.5))
    (tmp_path / "c.toml").write_text(s.to_toml(), encoding="utf-8")
    assert cli.load_config(tmp_path / "c.toml") == s.to_params()


def test_the_config_writer_refuses_non_finite_numbers():
    s = form.FormState(folder="db", event_type="spindle", output_path="out",
                       comparison="pairedT", sessions=["ses-1", "ses-2"],
                       band=(float("nan"), 15.0))
    with pytest.raises(ValueError, match="cannot write nan"):
        s.to_toml()


def test_choices_list_joint_stage_tokens_by_stage(tmp_path):
    make_db(tmp_path / "s01" / "wonambi", [("E1", 1.0, 40.0), ("E2", 1.5, 30.0)])
    c = form.choices(tw.available(tw.find_recordings(tmp_path)), "spindle")
    assert c == {"methods": ["Moelle2011"], "stages": ["NREM2", "NREM3"],
                 "bands": [(12.0, 15.0)]}


def test_a_run_stored_without_a_band_offers_no_band(tmp_path):
    db = make_db(tmp_path / "s01" / "wonambi", [("E1", 1.0, 40.0)])
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO events VALUES ('E1', 'slow_wave', 'Massimini2004', 'NREM3', "
                    "NULL, NULL, 1.0, 50.0)")
    c = form.choices(tw.available(tw.find_recordings(tmp_path)), "slow_wave")
    assert c == {"methods": ["Massimini2004"], "stages": ["NREM3"], "bands": []}
    s = form.FormState(folder=str(tmp_path), event_type="slow_wave", band=None)
    assert "freq_lower" not in s.to_params()["turtlewave"]


def test_subjects_file_columns_exclude_the_subject_column(tmp_path):
    subjects_csv(tmp_path / "s.csv", group=["A", "B"] * 6)
    table = form.subjects_table(tmp_path / "s.csv")
    assert form.info_columns(table) == ["group"] and list(table["Subject"]) == SUBJECTS


def test_core_and_form_never_import_pyqt5():
    code = ("import sys, hdeeg_snpm.cli, hdeeg_snpm.output, hdeeg_snpm.gui.form; "
            "sys.exit(any(m.startswith('PyQt5') for m in sys.modules))")
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_gui_without_pyqt5_prints_how_to_install(monkeypatch, capsys):
    for name in ["PyQt5", *(m for m in sys.modules if m.startswith("PyQt5."))]:
        monkeypatch.setitem(sys.modules, name, None)      # as if PyQt5 were not installed
    monkeypatch.delitem(sys.modules, "hdeeg_snpm.gui.window", raising=False)
    monkeypatch.delattr(gui, "window", raising=False)      # else `from . import window` finds it
    assert gui.main() == 1
    assert 'pip install "hdeeg-snpm[gui]"' in capsys.readouterr().err
    assert cli.main(["gui"]) == 1


def test_the_reject_set_shown_is_the_one_detection_stored(tmp_path):
    def recs(name, *analysed):
        for i, a in enumerate(analysed):
            make_db(tmp_path / name / f"s0{i}" / "wonambi", [("E1", 1.0, 40.0)], f"s0{i}",
                    analysed=a)
        return tw.find_recordings(tmp_path / name)
    seconds = {"NREM2": 600.0}
    assert form.reject_text(recs("a", {(1, 0): seconds}, {"Artefact": seconds})) == \
        "Artefact (as detected)"
    assert form.reject_text(recs("b", {(1, 1): seconds}, {(1, 0): seconds})) == \
        "no reject set is stored by every database (Arousal, Artefact: s00; Artefact: s01); " \
        "these densities cannot be compared. Redetect so the runs match."
    assert form.reject_text(recs("c", None)) == "no analysed time stored, so no density"
    assert form.reject_text(recs("d", {(1, 0): seconds, (0, 0): seconds})) == \
        "several reject sets stored: nothing rejected and Artefact (s00)"
    assert form.reject_text(recs("e", {"Artefact,Arousal": seconds})).startswith(
        "could not read the reject sets of the databases: ")     # not TurtleWave's spelling
    assert form.reject_text(recs("f", {(1, 0): seconds}, {"Artefact,Arousal": seconds})) == \
        "differs between databases: Artefact (s00); unreadable (s01)"
    unshared = recs("g", {(1, 0): seconds, (1, 1): seconds}, {(0, 0): seconds, (0, 1): seconds})
    assert form.reject_unshared(unshared) and form.reject_choices(unshared) == []
    assert form.reject_text(unshared) == (
        "no reject set is stored by every database (several reject sets stored: Arousal, "
        "Artefact and Artefact: s00; several reject sets stored: nothing rejected and Arousal: "
        "s01); these densities cannot be compared. Redetect so the runs match.")
    with pytest.raises(ValueError) as err:
        tw.event_table(unshared, "density", "spindle", stage="NREM2", reject_types="Artefact")
    assert "only for the reject set(s) nothing rejected; Arousal" in str(err.value)
    with pytest.raises(ValueError) as err:
        tw.event_table(recs("h", {(1, 0): seconds}, {(1, 1): seconds}), "density", "spindle",
                       stage="NREM2")
    assert form.window_message(str(err.value)).endswith(
        "No reject set is stored by every database, so these densities cannot be compared. "
        "Redetect so the runs match.")


def test_a_choice_of_reject_set_when_a_database_stores_several(tmp_path):
    seconds = {"NREM2": 600.0}
    make_db(tmp_path / "db" / "s01" / "wonambi", [("E1", 1.0, 40.0)], "s01",
            analysed={"Artefact": seconds, "": seconds, "Arousal,Artefact": seconds})
    make_db(tmp_path / "db" / "s02" / "wonambi", [("E1", 1.0, 40.0)], "s02",
            analysed={"Artefact": seconds, "": seconds})
    recs = tw.find_recordings(tmp_path / "db")
    assert form.reject_choices(recs) == ["", "Artefact"]          # stored by every database
    assert form.reject_choices(recs[1:2]) == ["", "Artefact"]
    single = tmp_path / "one" / "s01" / "wonambi"
    make_db(single, [("E1", 1.0, 40.0)], "s01", analysed={(1, 0): seconds})
    assert form.reject_choices(tw.find_recordings(tmp_path / "one")) == []
    state = form.FormState(folder=str(tmp_path / "db"), event_type="spindle", measure="density",
                           comparison="pairedT", sessions=["", "x"], output_path="out",
                           reject_sets_found=["", "Artefact"])
    assert "the reject set of density" in state.missing()
    assert "reject_types" not in state.to_params()["turtlewave"]
    state.reject_types = ""
    assert "the reject set of density" not in state.missing()
    assert state.to_params()["turtlewave"]["reject_types"] == "none"
    assert 'reject_types = "none"' in state.to_toml()
    state.measure = "amplitude"
    assert "reject_types" not in state.to_params()["turtlewave"]
    blocked = form.FormState(folder="db", event_type="spindle", measure="density",
                             comparison="pairedT", sessions=["", "x"], output_path="out",
                             reject_unshared=True)
    assert any("share a reject set" in m for m in blocked.missing())
    with pytest.raises(ValueError) as err:
        tw.event_table(recs, "density", "spindle", stage="NREM2")
    text = form.window_message(str(err.value))
    assert 'name the one to use in "Rejected before density"' in text and "[turtlewave]" not in text


def test_a_subject_id_error_is_worded_for_the_window(tmp_path):
    make_db(tmp_path / "11xx" / "wonambi", [("E1", 1.0, 40.0)], subject="sub-99zz")
    with pytest.raises(ValueError) as err:
        tw.find_recordings(tmp_path)
    text = form.window_message(str(err.value))
    assert "set Subject ID from to folder or database" in text and "subject_from" not in text
