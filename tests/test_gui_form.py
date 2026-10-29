"""The window's form without Qt: the parameters it builds equal a hand-written config (with a
`[turtlewave]` section, or naming table files), its saved config reads back the same, and a run
from it equals `snpm run`."""

import shutil
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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
    "ancova": 'comparison = "ancova"\ngroup_col = "group"\ncovariate_cols = ["age"]',
    "regression": 'comparison = "regression"\npredictor_col = "score"\ncovariate_cols = ["age"]',
    "rmanova": 'comparison = "rmanova"',
    "mixed2way": 'comparison = "mixed2way"\ngroup_col = "group"',
    "correlationP": 'comparison = "correlationP"\nbehavior_col = "score"',
    "correlationS": 'comparison = "correlationS"\nbehavior_col = "score"',
}
ROLES = {
    "pairedT": {"sessions": ["ses-1", "ses-2"]},
    "unpairedT": {"group_col": "group", "groups": ["A", "B"]},
    "anova1": {"group_col": "group"},
    "ancova": {"group_col": "group", "covariate_cols": ["age"]},
    "regression": {"predictor_col": "score", "covariate_cols": ["age"]},
    "rmanova": {},
    "mixed2way": {"group_col": "group"},
    "correlationP": {"behavior_col": "score"},
    "correlationS": {"behavior_col": "score"},
}


def setup(tmp_path, comparison, out="out"):
    """The synthetic study, the form filled for ``comparison`` and the hand-written config."""
    paired = comparison in ("pairedT", "rmanova", "mixed2way")     # sessions are compared
    no_subjects = comparison in ("pairedT", "rmanova")
    study(tmp_path / "db", sessions=("ses-1", "ses-2") if paired else ("",))
    groups = ["A", "B", "C"] * 4 if comparison == "anova1" else ["A", "B"] * 6
    subjects_csv(tmp_path / "subjects.csv", group=groups, score=np.linspace(0, 1, 12),
                 age=np.arange(20, 32) % 7)
    state = form.FormState(
        folder=str(tmp_path / "db"), sessions_found=["ses-1", "ses-2"] if paired else [""],
        event_type="spindle", method="Moelle2011", stage=["NREM2", "NREM3"], band=(12.0, 15.0),
        measure="count", comparison=comparison, permutations=100, seed=7,
        subjects_file="" if no_subjects else str(tmp_path / "subjects.csv"),
        output_path=str(tmp_path / out), **ROLES[comparison])
    text = TOP[comparison] + '\nchannels = "egi"\ndatatype = "absolute"\npermutations = 100\n' \
        f'seed = 7\noutput_path = "{out}"\n\n[turtlewave]\nfolder = "db"\n' \
        + ("" if no_subjects else 'subjects_file = "subjects.csv"\n') + \
        'event_type = "spindle"\nmethod = "Moelle2011"\nstage = ["NREM2", "NREM3"]\n' \
        'freq_lower = 12\nfreq_upper = 15\nmeasure = "count"\nsubject_from = "auto"\n'
    (tmp_path / "hand.toml").write_text(text)
    return state, tmp_path / "hand.toml"


@pytest.mark.parametrize("comparison", form.SOURCE_ANALYSES["turtlewave"])
def test_params_equal_the_hand_written_config_and_read_back(tmp_path, comparison):
    state, hand = setup(tmp_path, comparison)
    assert state.missing() == []
    assert state.to_params() == cli.load_config(hand)
    saved = tmp_path / "saved.toml"
    saved.write_text(state.to_toml())
    assert cli.load_config(saved) == state.to_params()


@pytest.mark.parametrize("comparison", form.SOURCE_ANALYSES["turtlewave"])
def test_a_run_from_the_form_equals_snpm_run(tmp_path, comparison):
    state, hand = setup(tmp_path, comparison, out="gui_out")
    hand.write_text(hand.read_text().replace('"gui_out"', '"out"'))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = run_and_write(state.to_params())
    same(np.load(paths["npz"]), run(hand)[0])


EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "data"
DESIGN_META = ["Subject", "group", "logAHI", "age", "sex"]
CIRC_FILES = {"groupA.csv": "circ_phase_groupA.csv", "groupB.csv": "circ_phase_groupB.csv",
              "countsA.csv": "circ_phase_countsA.csv", "countsB.csv": "circ_phase_countsB.csv",
              "angles.csv": "circ_anglinear_angles.csv",
              "measure.csv": "circ_anglinear_measure.csv", "mixed.csv": "glm_mixed2way_2x3.csv"}
ANGLES = 'circ_units = "rad"\ncirc_convention = "literature_uppeak0"'
MIXED_META = 'meta_cols = ["Subject", "group", "condition"]'
CIRC_GROUP = (('data1_file = "groupA.csv"\ndata2_file = "groupB.csv"\n'
               f'count1_file = "countsA.csv"\ncount2_file = "countsB.csv"\n{ANGLES}'),
              {"data1_file": "groupA.csv", "data2_file": "groupB.csv",
               "data3_file": "countsA.csv", "data4_file": "countsB.csv", "circ_units": "rad",
               "circ_convention": "literature_uppeak0"})
TABLE_TOP = {
    "pairedT": ('tail = "both"\ndata1_file = "condA.csv"\ndata2_file = "condB.csv"',
                {"data1_file": "condA.csv", "data2_file": "condB.csv"}),
    "unpairedT": ('tail = "both"\ndata1_file = "condA.csv"\ndata2_file = "condB.csv"',
                  {"data1_file": "condA.csv", "data2_file": "condB.csv"}),
    "onesampleT": ('data1_file = "condA.csv"', {"data1_file": "condA.csv"}),
    "anova1": ('group_col = "group"\ndata_file = "design.csv"\n'
               f"meta_cols = {DESIGN_META!r}".replace("'", '"'),
               {"data1_file": "design.csv", "group_col": "group"}),
    "ancova": ('group_col = "group"\ncovariate_cols = ["age", "sex"]\ndata_file = "design.csv"\n'
               f"meta_cols = {DESIGN_META!r}".replace("'", '"'),
               {"data1_file": "design.csv", "group_col": "group",
                "covariate_cols": ["age", "sex"]}),
    "regression": ('predictor_col = "logAHI"\ncovariate_cols = ["age", "sex"]\n'
                   'data_file = "design.csv"\n'
                   f"meta_cols = {DESIGN_META!r}".replace("'", '"'),
                   {"data1_file": "design.csv", "predictor_col": "logAHI",
                    "covariate_cols": ["age", "sex"]}),
    "rmanova": (('subject_col = "Subject"\ncondition_col = "condition"\n'
                 f'data_file = "mixed.csv"\n{MIXED_META}'),
                {"data1_file": "mixed.csv", "subject_col": "Subject",
                 "condition_col": "condition"}),
    "mixed2way": (('subject_col = "Subject"\ngroup_col = "group"\ncondition_col = "condition"\n'
                   f'data_file = "mixed.csv"\n{MIXED_META}'),
                  {"data1_file": "mixed.csv", "subject_col": "Subject", "group_col": "group",
                   "condition_col": "condition"}),
    "correlationP": ('data1_file = "condA.csv"\ndata2_file = "scores.csv"',
                     {"data1_file": "condA.csv", "data2_file": "scores.csv"}),
    "correlationS": ('data1_file = "condA.csv"\ndata2_file = "scores.csv"',
                     {"data1_file": "condA.csv", "data2_file": "scores.csv"}),
    "circ_phase_group": CIRC_GROUP,
    "circ_phase_group_u2": CIRC_GROUP,
    "circ_corrAngLinear": ((f'measure_col = "measure"\n{ANGLES}\n'
                            'data1_file = "angles.csv"\ndata2_file = "measure.csv"'),
                           {"data1_file": "angles.csv", "data2_file": "measure.csv",
                            "measure_col": "measure", "circ_units": "rad",
                            "circ_convention": "literature_uppeak0"}),
}


def table_study(root):
    """Synthetic tables: the paired example (24 subjects x 256 EGI channels), a design table of
    the first condition with group, log AHI, age and sex, its Excel copy behind a notes sheet,
    a score table, and the shipped mixed-design and circular examples."""
    root.mkdir(exist_ok=True)
    for name, example in CIRC_FILES.items():
        shutil.copy(EXAMPLES / example, root / name)
    shutil.copy(EXAMPLES / "paired_condA.csv", root / "condA.csv")
    shutil.copy(EXAMPLES / "paired_condB.csv", root / "condB.csv")
    eeg = pd.read_csv(EXAMPLES / "paired_condA.csv")
    rng = np.random.default_rng(3)
    n = len(eeg)
    info = pd.DataFrame({"Subject": eeg["Subject"], "group": ["A", "B", "C"] * (n // 3),
                         "logAHI": np.round(rng.normal(1.2, 0.4, n), 3),
                         "age": rng.integers(30, 70, n), "sex": ["F", "M"] * (n // 2)})
    design = pd.concat([info, eeg.drop(columns="Subject")], axis=1)
    design.to_csv(root / "design.csv", index=False)
    with pd.ExcelWriter(root / "design.xlsx") as book:
        pd.DataFrame({"note": ["synthetic"]}).to_excel(book, sheet_name="notes", index=False)
        design.to_excel(book, sheet_name="data", index=False)
    info[["Subject", "logAHI"]].to_csv(root / "scores.csv", index=False)
    return root


def table_setup(tmp_path, comparison, out="out"):
    """The form filled for ``comparison`` from the synthetic tables, and the hand-written
    config of the same run."""
    table_study(tmp_path)
    top, fields = TABLE_TOP[comparison]
    fields = dict(fields)
    for k in ("data1_file", "data2_file", "data3_file", "data4_file"):
        if k in fields:
            fields[k] = str(tmp_path / fields[k])
    meta = form.other_columns(form.table_columns(fields["data1_file"]))
    state = form.FormState(source="table", comparison=comparison, permutations=100, seed=7,
                           meta_cols=meta, output_path=str(tmp_path / out), **fields)
    datatype = "" if comparison in form.CIRC else 'datatype = "absolute"\n'
    (tmp_path / "hand.toml").write_text(
        f'comparison = "{comparison}"\n{top}\nchannels = "egi"\n{datatype}'
        f'permutations = 100\nseed = 7\noutput_path = "{out}"\n')
    return state, tmp_path / "hand.toml"


@pytest.mark.parametrize("comparison", form.SOURCE_ANALYSES["table"])
def test_table_params_equal_the_hand_written_config_and_read_back(tmp_path, comparison):
    state, hand = table_setup(tmp_path, comparison)
    assert state.missing() == []
    assert state.to_params() == cli.load_config(hand)
    saved = tmp_path / "saved.toml"
    saved.write_text(state.to_toml())
    assert "[turtlewave]" not in saved.read_text()
    assert cli.load_config(saved) == state.to_params()


def test_the_detected_other_columns_are_the_meta_columns(tmp_path):
    table_study(tmp_path)
    cols = form.table_columns(tmp_path / "design.csv")
    assert form.other_columns(cols) == DESIGN_META
    assert form.role_columns(cols) == DESIGN_META[1:]
    assert form.table_caption(cols, "egi") == (
        "Detected: 256 channels, 5 other columns: Subject, group, logAHI, age, sex. "
        "177 of them are EGI 256 (HydroCel) channels.")
    assert form.table_caption(form.table_columns(tmp_path / "scores.csv"), "egi") == \
        "Detected: 0 channels, 2 other columns: Subject, logAHI."


def test_an_excel_sheet_is_named_in_the_config(tmp_path):
    state, hand = table_setup(tmp_path, "regression")
    assert form.sheet_names(tmp_path / "design.xlsx") == ["notes", "data"]
    assert form.sheet_names(tmp_path / "design.csv") == []
    assert form.table_columns(tmp_path / "design.xlsx", "notes") == ["note"]
    state.data1_file, state.data1_sheet = str(tmp_path / "design.xlsx"), "data"
    hand.write_text(hand.read_text().replace('data_file = "design.csv"',
                                             'data_file = "design.xlsx"\ndata_sheet = "data"'))
    assert state.to_params() == cli.load_config(hand)


@pytest.mark.parametrize("comparison", ["regression", "ancova"])
def test_a_table_run_from_the_form_equals_snpm_run(tmp_path, comparison):
    state, hand = table_setup(tmp_path, comparison, out="gui_out")
    hand.write_text(hand.read_text().replace('"gui_out"', '"out"'))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = run_and_write(state.to_params())
    npz, js = run(hand)
    same(np.load(paths["npz"]), npz)
    assert js["params"]["covariate_cols"] == ["age", "sex"]


@pytest.mark.parametrize("comparison", ["rmanova", "mixed2way", "circ_corrAngLinear",
                                        "circ_phase_group"])
def test_a_new_table_analysis_from_the_form_equals_snpm_run(tmp_path, comparison):
    state, hand = table_setup(tmp_path, comparison, out="gui_out")
    hand.write_text(hand.read_text().replace('"gui_out"', '"out"'))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = run_and_write(state.to_params())
    same(np.load(paths["npz"]), run(hand)[0])


def test_a_circular_test_has_no_tail_or_datatype_and_needs_units_and_zero(tmp_path):
    state, hand = table_setup(tmp_path, "circ_phase_group")
    params = state.to_params()
    assert "tail" not in params and "datatype" not in params
    assert "circ_zero_offset_deg" not in params
    state.circ_units = state.circ_convention = ""
    assert state.roles() == ["circ_units", "circ_convention"]
    assert state.missing() == ["the angle units", "where zero sits in the angles"]
    state.circ_units, state.circ_convention = "deg", "custom"
    assert state.roles() == ["circ_units", "circ_convention", "circ_zero_offset_deg"]
    assert state.missing() == ["the zero offset in degrees"]
    state.circ_zero_offset_deg = -90.5
    assert state.missing() == []
    hand.write_text(hand.read_text().replace(ANGLES, 'circ_units = "deg"\n'
                                             'circ_convention = "custom"\n'
                                             'circ_zero_offset_deg = -90.5'))
    assert state.to_params() == cli.load_config(hand)
    state.circ_convention = "luna_zerocross0"             # the offset is for custom only
    assert "circ_zero_offset_deg" not in state.to_params()
    assert tuple(form.ZERO_CONVENTIONS) == ("literature_uppeak0", "yasa_uppeak0", "luna_zerocross0",
                                     "turtlewave_pre_v4", "custom")


def test_count_files_and_their_sheets_are_named_in_the_config(tmp_path):
    state, hand = table_setup(tmp_path, "circ_phase_group_u2")
    with pd.ExcelWriter(tmp_path / "countsA.xlsx") as book:
        pd.read_csv(tmp_path / "countsA.csv").to_excel(book, sheet_name="counts", index=False)
    state.data3_file, state.data3_sheet = str(tmp_path / "countsA.xlsx"), "counts"
    hand.write_text(hand.read_text().replace('count1_file = "countsA.csv"',
                                             'count1_file = "countsA.xlsx"\n'
                                             'count1_sheet = "counts"'))
    assert state.to_params() == cli.load_config(hand)
    state.data4_file = ""
    assert state.missing() == ["a group 2 event-count file (press Enter or Load to read it)"]
    assert "count2_file" not in state.to_params()


def test_the_measure_menu_lists_the_measure_files_columns(tmp_path):
    table_study(tmp_path)
    assert form.measure_choices(form.table_columns(tmp_path / "measure.csv")) == ["measure"]
    assert form.measure_choices(form.table_columns(tmp_path / "scores.csv")) == ["logAHI"]
    state, _ = table_setup(tmp_path, "circ_corrAngLinear")
    state.measure_col = ""
    assert state.missing() == ["a measure column"]


def test_mixed_designs_from_databases_take_the_session_as_condition():
    s = form.FormState(folder="db", event_type="spindle", output_path="out", comparison="rmanova",
                       sessions_found=["ses-1", "ses-2"])
    assert s.roles() == [] and s.missing() == []
    assert not {"subject_col", "condition_col", "sessions"} & set(s.to_params())
    s.comparison = "mixed2way"
    assert s.roles() == ["subjects_file", "group_col"]
    assert s.missing() == ["a subjects file (press Enter or Load to read it)", "a group column"]
    assert not set(form.CIRC) & set(form.SOURCE_ANALYSES["turtlewave"])
    assert set(form.CIRC) <= set(form.SOURCE_ANALYSES["table"])


def test_table_missing_lists_files_and_roles():
    s = form.FormState(source="table", comparison="pairedT")
    assert s.roles() == [] and s.files() == ("Condition 1 file", "Condition 2 file")
    assert s.missing() == ["a condition 1 file (press Enter or Load to read it)",
                           "a condition 2 file (press Enter or Load to read it)",
                           "an output folder"]
    s = form.FormState(source="table", comparison="regression", data1_file="d.csv",
                       output_path="out", sessions_found=["a", "b"])
    assert s.roles() == ["predictor_col", "covariate_cols"]
    assert s.missing() == ["a predictor column"]
    s.predictor_col = "logAHI"
    assert s.missing() == [] and "covariate_cols" not in s.to_params()
    s.comparison = "ancova"
    assert s.missing() == ["a group column", "at least one covariate column"]
    s.comparison = "correlationS"
    assert s.missing() == ["a score file (press Enter or Load to read it)"]
    s.comparison = "mixed2way"
    assert s.roles() == ["subject_col", "group_col", "condition_col"]
    assert s.missing() == ["a subject column", "a group column", "a condition column"]


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


@pytest.mark.parametrize("ext", ["csv", "xlsx"])
def test_a_numeric_header_is_a_column_name(tmp_path, ext):
    from hdeeg_snpm.io import tables
    from hdeeg_snpm.pipeline import prepare_glm
    table_study(tmp_path)
    design = pd.read_csv(tmp_path / "design.csv").rename(columns={"age": 1995})
    path = tmp_path / f"numeric.{ext}"
    design.to_csv(path, index=False) if ext == "csv" else design.to_excel(path, index=False)
    cols = form.table_columns(path)
    assert "1995" in cols and all(isinstance(c, str) for c in tables.read_table(path).columns)
    meta = form.other_columns(cols)
    assert meta == ["Subject", "group", "logAHI", "1995", "sex"]
    _, info, roles, _ = prepare_glm({"comparison": "regression", "data_file": str(path),
                                     "meta_cols": meta, "predictor_col": "logAHI",
                                     "covariate_cols": ["1995"]})
    assert roles["covariate_cols"] == ["1995"] and list(info["1995"]) == list(design[1995])


def test_the_predictor_menu_offers_only_numeric_columns(tmp_path):
    """A regression predictor must be numeric (the pipeline refuses a text one), so the window
    offers only the columns read as numbers; covariates may be text (dummy-coded)."""
    cols, numeric = form.table_info(EXAMPLES / "glm_regression.csv")
    assert numeric[:2] == ["logAHI", "age"] and "group" not in numeric and "sex" not in numeric
    assert form.predictor_choices(form.role_columns(cols), numeric) == ["logAHI", "age"]
    assert form.covariate_choices(form.role_columns(cols), "regression", "", "logAHI") == \
        ["group", "age", "sex"]
    path = tmp_path / "s.csv"
    pd.DataFrame({"Subject": ["01", "02", "03"], "group": ["A", "B", "A"],
                  "score": [0.1, None, 0.3], "age": [30, 40, 50],
                  "flag": [True, False, True]}).to_csv(path, index=False)
    assert form.subjects_numeric_columns(path) == ["score", "age"]      # true/false left out
    assert form.numeric_columns(pd.DataFrame({"f": [True, False], "x": [1.0, 2.0]})) == ["x"]
