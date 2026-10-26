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
