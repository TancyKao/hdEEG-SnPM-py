"""`snpm run` with a [turtlewave] section: synthetic databases to results in one command, and
equal to building the tables first and naming them in a hand-written config."""

import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import cli
from hdeeg_snpm.io import turtlewave as tw
from hdeeg_snpm.io import turtlewave_config as twc
from hdeeg_snpm.montage import get_montage
from test_turtlewave import add_turtlewave_tables, make_db, make_runs

CHANNELS = list(get_montage("egi").labels[:40])
SCOPE = 'event_type = "spindle"\nmethod = "Moelle2011"\nstage = ["NREM2", "NREM3"]\n' \
        "freq_lower = 12\nfreq_upper = 15\n"
SUBJECTS = [f"s{i:02d}" for i in range(1, 13)]


def study(root, subjects=SUBJECTS, sessions=("",), seed=0, skip=()):
    """One database per subject and session, events on 40 EGI channels; ``skip`` lists
    (subject, session) pairs with no database."""
    rng = np.random.default_rng(seed)
    for s in subjects:
        for ses in sessions:
            if (s, ses) in skip:
                continue
            events = [(ch, float(rng.uniform(0.5, 2)), float(rng.normal(40, 8)))
                      for ch in CHANNELS for _ in range(int(rng.integers(1, 8)))]
            make_db(root / s / ses / "wonambi" if ses else root / s / "wonambi", events)
    return root


def subjects_csv(path, subjects=SUBJECTS, **cols):
    pd.DataFrame({"Subject": subjects, **cols}).to_csv(path, index=False)
    return path


def config(path, top, measure="count", subjects_file=None):
    text = top + '\nchannels = "egi"\npermutations = 100\nseed = 7\noutput_path = "out"\n' \
        f'\n[turtlewave]\nfolder = "db"\nmeasure = "{measure}"\n' + SCOPE
    if subjects_file:
        text += f'subjects_file = "{subjects_file}"\n'
    path.write_text(text)
    return path


def run(cfg):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert cli.main(["run", str(cfg)]) == 0
    out = cfg.parent / "out"
    js = max(out.glob("*.json"), key=lambda p: p.stat().st_mtime)
    return np.load(js.with_suffix(".npz")), json.loads(js.read_text())


def same(a, b):
    assert set(a.files) == set(b.files)
    for k in a.files:
        assert np.array_equal(a[k], b[k], equal_nan=True), k


def two_step(tmp_path, top, split):
    """The route without the section: write_tables, then files split by hand, then a config."""
    paths = tw.write_tables(tmp_path / "db", tmp_path / "manual", "spindle", measures=("count",),
                            method="Moelle2011", stage=["NREM2", "NREM3"], freq_lower=12,
                            freq_upper=15)
    names = split(pd.read_csv(paths["count"]), tmp_path / "manual")
    hand = tmp_path / "hand"
    hand.mkdir()
    lines = "".join(f'{k} = "{v}"\n' for k, v in names.items() if k != "meta_cols")
    if "meta_cols" in names:
        lines += f"meta_cols = {json.dumps(names['meta_cols'])}\n"
    (hand / "run.toml").write_text(top + "\n" + lines + 'channels = "egi"\npermutations = 100\n'
                                   'seed = 7\noutput_path = "out"\n')
    return run(hand / "run.toml")[0]


def test_unpaired_from_databases_equals_the_two_step_route(tmp_path):
    study(tmp_path / "db")
    groups = ["A", "B"] * 6
    subjects_csv(tmp_path / "subjects.csv", group=groups)
    top = 'comparison = "unpairedT"\ngroup_col = "group"\ngroups = ["A", "B"]'
    npz, js = run(config(tmp_path / "study.toml", top, subjects_file="subjects.csv"))
    assert js["source"]["n_databases"] == 12 and js["source"]["measure"] == "count"
    assert js["source"]["band"] == "12-15 Hz" and js["source"]["dropped_subjects"] == []
    assert len(list((tmp_path / "out").glob("*_unpairedT_*_tables/recordings.csv"))) == 1
    html = next((tmp_path / "out").glob("*_report.html")).read_text()
    assert "<h2>Source</h2>" in html and "Moelle2011" in html

    def split(t, d):
        g = np.array(groups)
        t = t.drop(columns="session")
        t[g == "A"].to_csv(d / "a.csv", index=False)
        t[g == "B"].to_csv(d / "b.csv", index=False)
        return {"data1_file": d / "a.csv", "data2_file": d / "b.csv"}
    same(npz, two_step(tmp_path, 'comparison = "unpairedT"', split))


def test_paired_drops_and_lists_a_subject_missing_a_session(tmp_path):
    study(tmp_path / "db", sessions=("ses-1", "ses-2"), skip={("s05", "ses-2")})
    top = 'comparison = "pairedT"\nsessions = ["ses-1", "ses-2"]'
    cfg = config(tmp_path / "study.toml", top)
    with pytest.warns(UserWarning, match="s05 \\(no database for session ses-2\\)"):
        twc.prepare(cli.load_config(cfg), tmp_path / "probe")
    npz, js = run(cfg)
    assert js["source"]["dropped_subjects"] == [
        {"subject": "s05", "reason": "no database for session ses-2"}]
    assert any("s05" in n for n in js["notes"])

    def split(t, d):
        t = t[t["Subject"] != "s05"]
        for ses, f in (("ses-1", "a.csv"), ("ses-2", "b.csv")):
            t[t["session"] == ses].drop(columns="session").to_csv(d / f, index=False)
        return {"data1_file": d / "a.csv", "data2_file": d / "b.csv"}
    same(npz, two_step(tmp_path, 'comparison = "pairedT"', split))


def test_anova1_from_databases_equals_the_two_step_route(tmp_path):
    study(tmp_path / "db")
    groups = ["A", "B", "C"] * 4
    subjects_csv(tmp_path / "subjects.csv", group=groups)
    top = 'comparison = "anova1"\ngroup_col = "group"'
    npz, js = run(config(tmp_path / "study.toml", top, subjects_file="subjects.csv"))
    assert js["params"]["meta_cols"] == ["Subject", "group"]

    def split(t, d):
        t = t.drop(columns="session")
        t.insert(1, "group", groups)
        t.to_csv(d / "design.csv", index=False)
        return {"data_file": d / "design.csv", "meta_cols": ["Subject", "group"]}
    same(npz, two_step(tmp_path, 'comparison = "anova1"\ngroup_col = "group"', split))


@pytest.mark.parametrize("top, cols, measure", [
    ('comparison = "onesampleT"', {}, "count"),
    ('comparison = "ancova"\ngroup_col = "group"\ncovariate_cols = ["age"]',
     {"group": ["A", "B"] * 6, "age": np.arange(20, 32)}, "amplitude"),
    ('comparison = "regression"\npredictor_col = "score"',
     {"score": np.linspace(0, 1, 12)}, "count"),
    ('comparison = "correlationP"', {"score": np.linspace(0, 1, 12)}, "amplitude"),
    ('comparison = "correlationS"\nbehavior_col = "score"',
     {"score": np.linspace(0, 1, 12), "age": np.arange(20, 32)}, "count"),
])
def test_one_row_per_subject_analyses_run_end_to_end(tmp_path, top, cols, measure):
    study(tmp_path / "db")
    sf = subjects_csv(tmp_path / "subjects.csv", **cols).name if cols else None
    npz, js = run(config(tmp_path / "study.toml", top, measure, sf))
    assert js["source"]["measure"] == measure and npz.files


@pytest.mark.parametrize("top, cols", [
    ('comparison = "rmanova"', {}),
    ('comparison = "mixed2way"\ngroup_col = "group"', {"group": ["A", "B"] * 6}),
])
def test_session_designs_use_the_session_as_condition(tmp_path, top, cols):
    study(tmp_path / "db", sessions=("ses-1", "ses-2", "ses-3"))
    sf = subjects_csv(tmp_path / "subjects.csv", **cols).name if cols else None
    npz, js = run(config(tmp_path / "study.toml", top, subjects_file=sf))
    assert js["params"]["condition_col"] == "session" and js["params"]["subject_col"] == "Subject"
    assert npz.files


def prepared(tmp_path, top, measure="count", subjects_file=None):
    params = cli.load_config(config(tmp_path / "study.toml", top, measure, subjects_file))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return twc.prepare(params, tmp_path / "probe")


def test_subjects_on_one_side_only_are_refused_with_the_list(tmp_path):
    study(tmp_path / "db", subjects=SUBJECTS[:11] + ["01ab"])
    subjects_csv(tmp_path / "subjects.csv", group=["A", "B"] * 6)
    top = 'comparison = "unpairedT"\ngroups = ["A", "B"]'
    with pytest.raises(ValueError, match="no row in subjects.csv: 01ab.*no database: s12"):
        prepared(tmp_path, top, subjects_file="subjects.csv")


def test_sub_prefix_and_case_do_not_matter_when_joining(tmp_path):
    study(tmp_path / "db")
    pd.DataFrame({"Subject": [f"sub-S{i:02d}" for i in range(1, 13)],
                  "group": ["A", "B"] * 6}).to_csv(tmp_path / "subjects.csv", index=False)
    params, _, _ = prepared(tmp_path, 'comparison = "unpairedT"\ngroups = ["A", "B"]',
                            subjects_file="subjects.csv")
    assert len(pd.read_csv(params["data1_file"])) == 6


def test_interpolated_channels_are_listed_in_the_source(tmp_path):
    study(tmp_path / "db", subjects=SUBJECTS[1:])
    make_db(tmp_path / "db" / "s01" / "wonambi", [(ch, 1.0, 40.0) for ch in CHANNELS],
            subject="s01", interpolated=[CHANNELS[3]])
    params, source, _ = prepared(tmp_path, 'comparison = "onesampleT"')
    assert source["interpolated_channels"] == {"s01": [CHANNELS[3]]}
    assert np.isnan(pd.read_csv(params["data1_file"]).loc[0, CHANNELS[3]])


def test_two_runs_into_one_folder_keep_their_own_tables(tmp_path):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", group=["A", "B"] * 6)
    _, first = run(config(tmp_path / "ab.toml", 'comparison = "unpairedT"\ngroups = ["A", "B"]',
                          subjects_file="subjects.csv"))
    before = pd.read_csv(first["params"]["data1_file"])
    time.sleep(1.1)                                              # a new timestamp
    _, second = run(config(tmp_path / "ba.toml", 'comparison = "unpairedT"\ngroups = ["B", "A"]',
                           subjects_file="subjects.csv"))
    assert first["source"]["tables"] != second["source"]["tables"]
    pd.testing.assert_frame_equal(pd.read_csv(first["params"]["data1_file"]), before)
    assert len(list((tmp_path / "out").glob("*_tables"))) == 2


def test_sessions_are_not_pooled_into_a_one_recording_analysis(tmp_path):
    """Group A recorded at ses-1 and group B at ses-2: session would be confounded with group."""
    for i, s in enumerate(SUBJECTS):
        study(tmp_path / "db", subjects=[s], sessions=("ses-1" if i % 2 == 0 else "ses-2",),
              seed=i)
    subjects_csv(tmp_path / "subjects.csv", group=["A", "B"] * 6)
    for top in ('comparison = "unpairedT"\ngroups = ["A", "B"]',
                'comparison = "anova1"\ngroup_col = "group"'):
        with pytest.raises(ValueError, match="more than one session \\(ses-1, ses-2\\)"):
            prepared(tmp_path, top, subjects_file="subjects.csv")


@pytest.mark.parametrize("folders, listed, ok", [
    ([str(i) for i in range(1, 13)], list(range(1, 13)), True),          # numbers both sides
    ([f"{i:02d}" for i in range(1, 13)], [f"{i:02d}" for i in range(1, 13)], True),
    ([f"{i:02d}" for i in range(1, 13)], list(range(1, 13)), False),     # 01 is not 1
])
def test_numeric_subject_ids_are_compared_as_text(tmp_path, folders, listed, ok):
    study(tmp_path / "db", subjects=folders)
    subjects_csv(tmp_path / "subjects.csv", subjects=listed, group=["A", "B"] * 6)
    top = 'comparison = "unpairedT"\ngroups = ["A", "B"]'
    if ok:
        params, _, _ = prepared(tmp_path, top, subjects_file="subjects.csv")
        assert len(pd.read_csv(params["data1_file"])) == 6
    else:
        with pytest.raises(ValueError, match="no row in subjects.csv: 01, 02"):
            prepared(tmp_path, top, subjects_file="subjects.csv")


@pytest.mark.parametrize("top, sessions", [
    ('comparison = "rmanova"', ("ses-1", "ses-2")),
    ('comparison = "correlationP"', ("",)),
])
def test_ids_that_read_as_one_number_are_refused(tmp_path, top, sessions):
    ids = ["01", "1", "02", "2", "007", "7"] + [f"s{i}" for i in range(6)]
    study(tmp_path / "db", subjects=ids, sessions=sessions)
    subjects_csv(tmp_path / "subjects.csv", subjects=ids, score=np.linspace(0, 1, 12))
    with pytest.raises(ValueError, match="ambiguous: 007/7, 01/1, 02/2"):
        prepared(tmp_path, top, subjects_file="subjects.csv")


@pytest.mark.parametrize("top, cols", [
    ('comparison = "anova1"\ngroup_col = "group"', {"group": ["A", "B", "C"] * 4}),
    ('comparison = "correlationP"', {"score": np.linspace(0, 1, 12) ** 2}),
])
def test_subjects_file_row_order_does_not_matter(tmp_path, top, cols):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", **cols)
    (tmp_path / "rev").mkdir()
    subjects_csv(tmp_path / "rev" / "subjects.csv", subjects=SUBJECTS[::-1],
                 **{k: np.asarray(v)[::-1] for k, v in cols.items()})
    a, _ = run(config(tmp_path / "a.toml", top, subjects_file="subjects.csv"))
    time.sleep(1.1)
    b, _ = run(config(tmp_path / "b.toml", top, subjects_file="rev/subjects.csv"))
    same(a, b)


def test_source_covers_only_the_analysed_recordings(tmp_path):
    study(tmp_path / "db", subjects=SUBJECTS[1:])
    make_db(tmp_path / "db" / "s01" / "wonambi", [(ch, 1.0, 40.0) for ch in CHANNELS],
            subject="s01", interpolated=[CHANNELS[3]])
    subjects_csv(tmp_path / "subjects.csv", group=["C"] + ["A", "B"] * 5 + ["A"])
    cfg = config(tmp_path / "study.toml", 'comparison = "unpairedT"\ngroups = ["A", "B"]',
                 subjects_file="subjects.csv")
    cfg.write_text(cfg.read_text().replace("freq_upper = 15\n", "").replace(
        'method = "Moelle2011"\n', ""))
    _, source, _ = twc.prepare(cli.load_config(cfg), tmp_path / "probe")
    assert source["n_databases"] == 11 and source["n_databases_found"] == 12
    assert source["interpolated_channels"] == {}                 # s01 is in group C
    assert source["method"] == "Moelle2011" and source["band"] == "12-15 Hz"


def test_routing_keys_need_the_section_and_covariates_need_a_file(tmp_path):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", score=np.linspace(0, 1, 12))
    with pytest.raises(ValueError, match="use_covariates needs covariate_file"):
        prepared(tmp_path, 'comparison = "correlationP"\nuse_covariates = true',
                 subjects_file="subjects.csv")
    tw.write_tables(tmp_path / "db", tmp_path / "t", "spindle", measures=("count",))
    t = pd.read_csv(tmp_path / "t" / "eventStat_count_spindle.csv").drop(columns="session")
    t.to_csv(tmp_path / "t" / "a.csv", index=False)
    for key in ('sessions = ["ses-1"]', 'groups = ["A", "B"]', 'behavior_col = "score"'):
        (tmp_path / "run.toml").write_text(f'comparison = "onesampleT"\ndata1_file = "t/a.csv"\n'
                                           f'{key}\npermutations = 100\n')
        assert cli.main(["run", str(tmp_path / "run.toml")]) == 1


def test_refusals(tmp_path):
    study(tmp_path / "db", sessions=("ses-1", "ses-2"))
    with pytest.raises(ValueError, match="needs sessions"):
        prepared(tmp_path, 'comparison = "pairedT"')
    with pytest.raises(ValueError, match="ses-3 are not among"):
        prepared(tmp_path, 'comparison = "pairedT"\nsessions = ["ses-1", "ses-3"]')
    with pytest.raises(ValueError, match="more than one session"):
        prepared(tmp_path, 'comparison = "onesampleT"')
    assert prepared(tmp_path, 'comparison = "onesampleT"\nsessions = ["ses-2"]')[0]["data1_file"]
    with pytest.raises(ValueError, match="remove data1_file"):
        prepared(tmp_path, 'comparison = "onesampleT"\ndata1_file = "x.csv"')
    for key in ("mixedmodel", "circ_phase_group"):
        with pytest.raises(ValueError, match=f"{key} cannot be run from a \\[turtlewave\\]"):
            prepared(tmp_path, f'comparison = "{key}"')
    with pytest.raises(ValueError, match="needs subjects_file"):
        prepared(tmp_path, 'comparison = "anova1"\ngroup_col = "group"')
    cfg = config(tmp_path / "bad.toml", 'comparison = "onesampleT"')
    cfg.write_text(cfg.read_text() + "drop = true\n")
    assert cli.main(["run", str(cfg)]) == 1                         # unknown key: refused


def test_runs_without_the_section_have_no_source(tmp_path):
    study(tmp_path / "db")
    tw.write_tables(tmp_path / "db", tmp_path / "t", "spindle", measures=("count",))
    t = pd.read_csv(tmp_path / "t" / "eventStat_count_spindle.csv").drop(columns="session")
    t.to_csv(tmp_path / "t" / "a.csv", index=False)
    (tmp_path / "run.toml").write_text('comparison = "onesampleT"\ndata1_file = "t/a.csv"\n'
                                       'permutations = 100\nseed = 1\noutput_path = "out"\n')
    _, js = run(tmp_path / "run.toml")
    assert "source" not in js
    assert "<h2>Source</h2>" not in next((tmp_path / "out").glob("*_report.html")).read_text()


def test_density_from_databases(tmp_path):
    """Density goes through TurtleWave; run in TurtleWave's environment."""
    pytest.importorskip("turtlewave_hdEEG.density")
    rng = np.random.default_rng(1)
    for s in SUBJECTS:
        events = [(ch, 1.0, 40.0) for ch in CHANNELS for _ in range(int(rng.integers(1, 8)))]
        path = make_runs(tmp_path / "db" / s / "wonambi",
                         [("spindle", "NREM2NREM3", 12, 15, events)])
        add_turtlewave_tables(path, s, {"NREM2": 1800.0, "NREM3": 600.0},
                              [(ch, "spindle", "m", 12, 15, "NREM2NREM3", 1) for ch in CHANNELS])
    cfg = config(tmp_path / "study.toml", 'comparison = "onesampleT"', "density")
    cfg.write_text(cfg.read_text().replace('"Moelle2011"', '"m"'))
    _, js = run(cfg)
    assert js["source"]["density_divided_by"] == ["NREM2+NREM3"]
    assert js["source"]["turtlewave_version"]


@pytest.mark.parametrize("labels, groups", [
    (["01", "02"] * 5 + ["01", ""], '["01", "02"]'),
    (["1", "2"] * 5 + ["1", ""], "[1, 2]"),
    (["1.5", "2.5"] * 5 + ["1.5", ""], '["1.5", "2.5"]'),
])
def test_group_labels_are_compared_as_written_and_an_empty_one_is_left_out(tmp_path, labels,
                                                                          groups):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", group=labels)
    top = f'comparison = "unpairedT"\ngroup_col = "group"\ngroups = {groups}'
    params, source, notes = prepared(tmp_path, top, subjects_file="subjects.csv")
    assert source["dropped_subjects"] == [{"subject": "s12", "reason": "group is empty"}]
    assert "s12 (empty)" in notes[0]
    assert len(pd.read_csv(params["data1_file"])) == 6
    assert len(pd.read_csv(params["data2_file"])) == 5


def test_a_group_label_written_differently_is_refused_with_the_labels_held(tmp_path):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", group=["01", "02"] * 6)
    top = 'comparison = "unpairedT"\ngroup_col = "group"\ngroups = [1, 2]'
    with pytest.raises(ValueError, match=r"Group\(s\) 1, 2 do not occur in 'group' \(01, 02\)"):
        prepared(tmp_path, top, subjects_file="subjects.csv")


def test_a_group_named_na_is_a_label_and_an_empty_group_name_is_refused(tmp_path):
    study(tmp_path / "db")
    subjects_csv(tmp_path / "subjects.csv", group=["NA", "None"] * 5 + ["NA", " "])
    top = 'comparison = "unpairedT"\ngroup_col = "group"\ngroups = ["NA", "None"]'
    params, source, _ = prepared(tmp_path, top, subjects_file="subjects.csv")
    assert source["dropped_subjects"] == [{"subject": "s12", "reason": "group is empty"}]
    assert len(pd.read_csv(params["data1_file"])) == 6
    with pytest.raises(ValueError, match="groups names an empty group"):
        prepared(tmp_path, top.replace('"None"]', '""]'), subjects_file="subjects.csv")
