"""TurtleWave database reader, on small databases built here with the columns the reader uses."""

import json
import sqlite3
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm.io import tables
from hdeeg_snpm.io import turtlewave as tw


def make_db(folder, events, subject=None, interpolated=None):
    """events: (channel, duration, peak2peak) rows of one spindle run stored as 'NREM2NREM3'."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / tw.DB_NAME
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (channel TEXT, event_type TEXT, method TEXT, stage TEXT, "
                "freq_lower REAL, freq_upper REAL, duration REAL, peak2peak_amp REAL)")
    con.executemany("INSERT INTO events VALUES (?, 'spindle', 'Moelle2011', 'NREM2NREM3', 12, 15, "
                    "?, ?)", events)
    if subject is not None:
        con.execute("CREATE TABLE stage_durations (subject TEXT)")
        con.execute("INSERT INTO stage_durations VALUES (?)", (subject,))
    if interpolated is not None:
        con.execute("CREATE TABLE detection_runs (subject TEXT, interpolated_channels TEXT)")
        con.execute("INSERT INTO detection_runs VALUES (?, ?)", (subject, json.dumps(interpolated)))
    con.commit()
    con.close()
    return path


EVENTS = [("E1", 1.0, 40.0), ("E1", 2.0, 60.0), ("E2", 1.5, 30.0), ("E10", 0.5, 20.0)]


def test_old_database_takes_subject_and_session_from_the_folders(tmp_path):
    make_db(tmp_path / "s01" / "ses-1" / "wonambi", EVENTS)
    make_db(tmp_path / "s01" / "ses-2" / "wonambi", EVENTS[:2])
    make_db(tmp_path / "s02" / "wonambi", EVENTS)
    recs = tw.find_recordings(tmp_path)
    assert [(r.subject, r.session, r.subject_source) for r in recs] == [
        ("s01", "ses-1", "folder"), ("s01", "ses-2", "folder"), ("s02", "", "folder")]


def test_new_database_subject_must_agree_with_the_folder(tmp_path):
    make_db(tmp_path / "01ab" / "wonambi", EVENTS, subject="sub-01ab")
    assert tw.find_recordings(tmp_path)[0].subject == "sub-01ab"      # 'sub-' prefix is ignored
    bad = make_db(tmp_path / "11xx" / "wonambi", EVENTS, subject="sub-99zz")
    with pytest.raises(ValueError, match="sub-99zz.*11xx"):
        tw.find_recordings(tmp_path)
    assert tw.find_recordings([bad], subject_from="database")[0].subject == "sub-99zz"
    assert tw.find_recordings([bad], subject_from="folder")[0].subject == "11xx"


def test_two_databases_for_one_subject_and_session_are_refused(tmp_path):
    a = make_db(tmp_path / "x" / "s01" / "wonambi", EVENTS)
    b = make_db(tmp_path / "y" / "s01" / "wonambi", EVENTS)
    with pytest.raises(ValueError, match="same subject and session"):
        tw.find_recordings([a, b])


def test_measures_per_channel(tmp_path):
    make_db(tmp_path / "s01" / "wonambi", EVENTS)
    make_db(tmp_path / "s02" / "wonambi", EVENTS[2:])
    recs = tw.find_recordings(tmp_path)
    amp, manifest = tw.event_table(recs, "amplitude", "spindle", stage=["NREM2", "NREM3"])
    assert list(amp.columns) == ["Subject", "session", "E1", "E2", "E10"]     # natural order
    assert amp.loc[0, "E1"] == 50.0 and np.isnan(amp.loc[1, "E1"])            # no event: missing
    dur, _ = tw.event_table(recs, "duration", "spindle")
    assert dur.loc[0, "E1"] == 1.5
    count, _ = tw.event_table(recs, "count", "spindle")
    assert count.loc[0, "E1"] == 2 and count.loc[1, "E1"] == 0                # no event: zero
    assert list(manifest["n_channels_with_events"]) == [3, 2]


def test_interpolated_channels_are_reported_and_set_to_missing(tmp_path):
    make_db(tmp_path / "s01" / "wonambi", EVENTS, subject="s01", interpolated=["E2"])
    recs = tw.find_recordings(tmp_path)
    amp, manifest = tw.event_table(recs, "amplitude", "spindle")
    assert np.isnan(amp.loc[0, "E2"]) and manifest.loc[0, "interpolated_channels"] == "E2"
    count, _ = tw.event_table(recs, "count", "spindle")
    assert np.isnan(count.loc[0, "E2"]) and count.loc[0, "E1"] == 2      # missing, not zero
    kept, _ = tw.event_table(recs, "amplitude", "spindle", drop_interpolated=False)
    assert kept.loc[0, "E2"] == 30.0


def test_a_request_matching_nothing_or_two_runs_is_an_error(tmp_path):
    path = make_db(tmp_path / "s01" / "wonambi", EVENTS)
    recs = tw.find_recordings(tmp_path)
    with pytest.raises(ValueError, match="no 'slow_wave' events"):
        tw.event_table(recs, "amplitude", "slow_wave")
    con = sqlite3.connect(path)
    con.execute("INSERT INTO events VALUES ('E1', 'spindle', 'Moelle2011', 'NREM2', 12, 15, 1, 1)")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="more than one method, stage or band"):
        tw.event_table(recs, "amplitude", "spindle")
    assert tw.event_table(recs, "count", "spindle", stage="NREM2")[0].loc[0, "E1"] == 1


def test_density_needs_turtlewave_and_says_so(tmp_path):
    path = make_db(tmp_path / "s01" / "wonambi", EVENTS)
    recs = tw.find_recordings(tmp_path)
    with pytest.raises(ValueError, match="no analysed time"):       # before TurtleWave 4.3
        tw.event_table(recs, "density", "spindle")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE analysed_time (subject TEXT, stage TEXT, analysed_seconds REAL)")
    con.commit()
    con.close()
    try:
        import turtlewave_hdEEG  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="turtlewave"):
            tw.event_table(recs, "density", "spindle")


def test_tables_are_written_with_a_recordings_file(tmp_path):
    make_db(tmp_path / "in" / "s01" / "ses-a" / "wonambi", EVENTS)
    paths = tw.write_tables(tmp_path / "in", tmp_path / "out", "spindle",
                            measures=("amplitude", "duration"))
    assert paths["amplitude"].name == "eventStat_amplitude_spindle.csv"
    assert "ses-a" in paths["recordings"].read_text()


def test_subject_folders_that_merely_start_with_ses_are_not_sessions(tmp_path):
    for name in ("SES01", "SES02"):
        make_db(tmp_path / "study" / name / "wonambi", EVENTS)
    recs = tw.find_recordings(tmp_path)
    assert [(r.subject, r.session) for r in recs] == [("SES01", ""), ("SES02", "")]
    make_db(tmp_path / "one" / "s01" / "ses-1", EVENTS)
    make_db(tmp_path / "one" / "s01" / "ses-2", EVENTS)
    with pytest.warns(UserWarning, match="assigned to one subject"):
        tw.find_recordings(tmp_path / "one")


def make_runs(folder, runs):
    """runs: (event_type, stage, freq_lower, freq_upper, [(channel, duration, peak2peak)])."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / tw.DB_NAME
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE events (channel TEXT, event_type TEXT, method TEXT, stage TEXT, "
                "freq_lower REAL, freq_upper REAL, duration REAL, peak2peak_amp REAL)")
    for etype, stage, lo, hi, events in runs:
        con.executemany("INSERT INTO events VALUES (?, ?, 'm', ?, ?, ?, ?, ?)",
                        [(ch, etype, stage, lo, hi, d, a) for ch, d, a in events])
    con.commit()
    con.close()
    return path


def test_stage_request_pools_per_stage_and_joint_storage_alike(tmp_path):
    joint = make_runs(tmp_path / "a" / "wonambi", [("spindle", "NREM2NREM3", 12, 15, EVENTS)])
    split = make_runs(tmp_path / "b" / "wonambi", [
        ("spindle", "NREM2", 12, 15, EVENTS[:2]), ("spindle", "NREM3", 12, 15, EVENTS[2:])])
    recs = tw.find_recordings([joint, split])
    for stage in (["NREM2", "NREM3"], "NREM2NREM3", ["NREM3", "NREM2"]):
        amp, _ = tw.event_table(recs, "amplitude", "spindle", stage=stage)
        assert amp.iloc[0, 2:].tolist() == amp.iloc[1, 2:].tolist() == [50.0, 30.0, 20.0]
    with pytest.raises(ValueError, match="more than one"):          # stage not named: no pooling
        tw.event_table(recs[1:], "amplitude", "spindle")
    assert tw.event_table(recs[1:], "count", "spindle", stage="NREM2")[0].loc[0, "E1"] == 2
    with pytest.raises(ValueError, match="no 'spindle' events"):    # a joint run cannot be split
        tw.event_table(recs[:1], "amplitude", "spindle", stage="NREM2")


def test_two_runs_covering_one_stage_are_not_pooled(tmp_path):
    path = make_runs(tmp_path / "a" / "wonambi", [
        ("spindle", "NREM2", 9, 12, EVENTS), ("spindle", "NREM2NREM3", 9, 12, EVENTS)])
    with pytest.raises(ValueError, match="more than one"):
        tw.event_table(tw.find_recordings([path]), "amplitude", "spindle",
                       stage=["NREM2", "NREM3"])


def add_turtlewave_tables(path, subject, seconds, roster):
    """The analysed_time table as TurtleWave >= 4.4 writes it (dbwrite.store_analysed_time),
    with seconds = {stage: analysed seconds} under the default reject set, and the
    processing_status table of eventprocessor.py; roster = (channel, event_type, method,
    freq_lower, freq_upper, stage, success) rows."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE analysed_time (
        subject TEXT NOT NULL, stage TEXT NOT NULL, reject_types TEXT NOT NULL,
        reject_artifacts INTEGER NOT NULL, reject_arousals INTEGER NOT NULL,
        analysed_seconds REAL NOT NULL, artefact_seconds_excluded REAL, epoch_length REAL,
        source TEXT, annotation_file TEXT, turtlewave_version TEXT, processing_timestamp TEXT,
        PRIMARY KEY (subject, stage, reject_types))""")
    con.executemany("INSERT INTO analysed_time VALUES (?, ?, 'Arousal,Artefact,Move', 1, 1, ?, "
                    "?, 30, 'detection', NULL, '4.6.2', NULL)",
                    [(subject, st, sec, 7.5) for st, sec in seconds.items()])
    con.execute("""CREATE TABLE processing_status (
        channel TEXT NOT NULL, event_type TEXT NOT NULL, method TEXT NOT NULL DEFAULT '',
        freq_lower REAL NOT NULL DEFAULT 0, freq_upper REAL NOT NULL DEFAULT 0,
        stage TEXT NOT NULL DEFAULT '', json_file TEXT, processed BOOLEAN DEFAULT 0,
        attempts INTEGER DEFAULT 0, last_attempt_time TEXT, success BOOLEAN DEFAULT 0,
        error_message TEXT,
        PRIMARY KEY (channel, event_type, method, freq_lower, freq_upper, stage))""")
    con.executemany("INSERT INTO processing_status (channel, event_type, method, freq_lower, "
                    "freq_upper, stage, processed, success) VALUES (?, ?, ?, ?, ?, ?, 1, ?)",
                    roster)
    con.commit()
    con.close()


def test_density_refuses_two_runs_covering_one_stage_before_turtlewave(tmp_path, monkeypatch):
    path = make_runs(tmp_path / "sub-01" / "wonambi", [
        ("spindle", "NREM2", 12, 15, EVENTS), ("spindle", "NREM2NREM3", 12, 15, EVENTS)])
    add_turtlewave_tables(path, "sub-01", {"NREM2": 600.0, "NREM3": 300.0}, [])
    monkeypatch.setitem(sys.modules, "turtlewave_hdEEG", None)     # importing it now fails
    with pytest.raises(ValueError, match="more than one method, stage or band"):
        tw.event_table(tw.find_recordings([path]), "density", "spindle",
                       stage=["NREM2", "NREM3"], freq_lower=12, freq_upper=15)


def test_density_zeros_come_from_the_selected_runs_method(tmp_path):
    pytest.importorskip("turtlewave_hdEEG.density")
    path = make_runs(tmp_path / "sub-01" / "wonambi", [("spindle", "NREM2NREM3", 12, 15, EVENTS)])
    add_turtlewave_tables(path, "sub-01", {"NREM2": 90.0, "NREM3": 30.0}, [
        *[(ch, "spindle", "m", 12, 15, "NREM2NREM3", 1) for ch in ("E1", "E2", "E5", "E10")],
        ("E7", "spindle", "other", 12, 15, "NREM2NREM3", 1)])
    dens, _ = tw.event_table(tw.find_recordings([path]), "density", "spindle",
                             stage=["NREM2", "NREM3"], freq_lower=12, freq_upper=15)
    assert dens.iloc[0, 2:].to_dict() == {"E1": 1.0, "E2": 0.5, "E5": 0.0, "E10": 0.5}


def test_a_folder_of_symlinks_names_subjects_by_the_links(tmp_path):
    real = make_db(tmp_path / "store" / "x" / "wonambi", EVENTS)
    link = tmp_path / "study" / "s07" / "wonambi" / tw.DB_NAME
    link.parent.mkdir(parents=True)
    link.symlink_to(real)
    assert tw.find_recordings(tmp_path / "study")[0].subject == "s07"


BRIDGE = Path(__file__).resolve().parents[2] / "SnPM_2025" / "db_to_group_table.py"


@pytest.mark.skipif(not BRIDGE.is_file(), reason="needs ../SnPM_2025/db_to_group_table.py")
def test_tables_equal_the_matlab_bridge_script(tmp_path):
    """Same synthetic databases, which have no analysed_time table (TurtleWave before 4.3),
    through db_to_group_table.py and the reader: amplitude and duration identical. The reader
    has no density for such a database; the bridge falls back to hypnogram minutes of the
    stages the selected events are stored under (only when analysed_time is absent and
    --allow-hypnogram-denominator is given), so its density times those minutes must give the
    reader's count. Every subject stores each event type under NREM2 and NREM3 (jointly or per
    stage), so all are read over the same stage set, as the bridge requires."""
    sw = [("E1", 1.0, 80.0), ("E2", 1.2, 75.5), ("E2", 0.9, 61.25), ("E129", 1.1, 90.0)]
    spec = {"s01": [("spindle", "NREM2NREM3", 12, 15, EVENTS),
                    ("slow_wave", "NREM2NREM3", 0.5, 1.25, sw)],
            "s02": [("spindle", "NREM2", 12, 15, EVENTS[1:3]),
                    ("spindle", "NREM3", 12, 15, EVENTS[3:]),
                    ("slow_wave", "NREM2", 0.5, 1.25, sw[:1]),
                    ("slow_wave", "NREM3", 0.5, 1.25, sw[1:])]}
    epochs_per_stage = {"s01": {"NREM2": 4, "NREM3": 2, "REM": 2},    # 30-s epochs
                        "s02": {"NREM2": 2, "NREM3": 1, "NREM1": 3}}
    minutes = {name: (n["NREM2"] + n["NREM3"]) / 2 for name, n in epochs_per_stage.items()}
    for name, runs in spec.items():
        make_runs(tmp_path / "in" / name / "wonambi", runs)
        stages = [st for st, n in epochs_per_stage[name].items() for _ in range(n)]
        epochs = "".join(f"<epoch><epoch_start>{30 * i}</epoch_start><epoch_end>{30 * i + 30}"
                         f"</epoch_end><stage>{st}</stage></epoch>" for i, st in enumerate(stages))
        (tmp_path / "in" / name / "wonambi" / f"{name}_clean_rebuilt.xml").write_text(
            f"<annotations><rater><stages>{epochs}</stages></rater></annotations>")
    done = subprocess.run([sys.executable, str(BRIDGE), str(tmp_path / "in"),
                           str(tmp_path / "out"), "--allow-hypnogram-denominator"],
                          check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr + done.stdout
    rec = pd.read_csv(tmp_path / "out" / "recordings.csv", dtype={"Subject": str})
    rec = rec[rec["band"].isin(["spindle_12-15", "sw_0.5-1.25"])]     # bands not detected: blank
    assert len(rec) == 4
    assert set(rec["stages"]) == {"NREM2+NREM3"}
    assert set(rec["density_denominator"]) == {"hypnogram"}
    for name, m in minutes.items():
        assert (rec.loc[rec["Subject"] == name, "denominator_minutes"] == m).all()
    recs = tw.find_recordings(tmp_path / "in")
    for etype, lo, hi, label in (("spindle", 12, 15, "spindle_12-15"),
                                 ("slow_wave", 0.5, 1.25, "sw_0.5-1.25")):
        def bridge(measure, label=label):
            t = tables.read_csv(tmp_path / "out" / f"eventStat_{measure}_{label}.csv",
                                dtype={"Subject": str})
            return t.set_index("Subject").dropna(axis=1, how="all")
        for measure in ("amplitude", "duration"):
            ours, _ = tw.event_table(recs, measure, etype, stage=["NREM2", "NREM3"],
                                     freq_lower=lo, freq_upper=hi)
            pd.testing.assert_frame_equal(ours.drop(columns="session").set_index("Subject"),
                                          bridge(measure), check_names=False)
        count, _ = tw.event_table(recs, "count", etype, stage=["NREM2", "NREM3"],
                                  freq_lower=lo, freq_upper=hi)
        count = count.drop(columns="session").set_index("Subject")
        implied = bridge("density").mul(pd.Series(minutes), axis=0)
        pd.testing.assert_frame_equal(count.where(implied.notna()), implied, check_names=False)


@pytest.mark.skipif(not BRIDGE.is_file(), reason="needs ../SnPM_2025/db_to_group_table.py")
def test_density_equals_the_matlab_bridge_script_with_analysed_time(tmp_path):
    """With an analysed_time table both divide by TurtleWave's artefact-free time of the stages
    the run covers, and both give 0 to a channel the selected run processed without an event
    and ignore channels only another method processed: the densities are equal exactly."""
    pytest.importorskip("turtlewave_hdEEG.density")
    sw = [("E1", 1.0, 80.0), ("E2", 1.2, 75.5), ("E2", 0.9, 61.25), ("E129", 1.1, 90.0)]
    spec = {"sub-01": ([("spindle", "NREM2NREM3", 12, 15, EVENTS),
                        ("slow_wave", "NREM2NREM3", 0.5, 1.25, sw)],
                       {"NREM1": 40.0, "NREM2": 1234.5, "NREM3": 987.25, "REM": 300.0}),
            "sub-02": ([("spindle", "NREM2", 12, 15, EVENTS[1:3]),
                        ("spindle", "NREM3", 12, 15, EVENTS[3:]),
                        ("slow_wave", "NREM3", 0.5, 1.25, sw[1:])],
                       {"NREM2": 1111.1, "NREM3": 333.3})}
    for name, (runs, seconds) in spec.items():
        path = make_runs(tmp_path / "in" / name / "wonambi", runs)
        roster = [(ch, et, "m", lo, hi, "NREM2NREM3", 1)
                  for et, lo, hi in (("spindle", 12, 15), ("slow_wave", 0.5, 1.25))
                  for ch in ("E1", "E2", "E5", "E10", "E129")]
        roster += [("E7", "spindle", "other", 12, 15, "NREM2NREM3", 1),
                   ("E8", "spindle", "m", 12, 15, "NREM2NREM3", 0)]
        add_turtlewave_tables(path, name, seconds, roster)
        (tmp_path / "in" / name / "wonambi" / f"{name}_clean_rebuilt.xml").write_text(
            "<annotations><rater><stages><epoch><epoch_start>0</epoch_start><epoch_end>30"
            "</epoch_end><stage>NREM2</stage></epoch></stages></rater></annotations>")
    subprocess.run([sys.executable, str(BRIDGE), str(tmp_path / "in"), str(tmp_path / "out")],
                   check=True, capture_output=True)
    recs = tw.find_recordings(tmp_path / "in")
    for etype, lo, hi, label in (("spindle", 12, 15, "spindle_12-15"),
                                 ("slow_wave", 0.5, 1.25, "sw_0.5-1.25")):
        bridge = tables.read_csv(tmp_path / "out" / f"eventStat_density_{label}.csv",
                                 dtype={"Subject": str})
        bridge = bridge.set_index("Subject").dropna(axis=1, how="all")
        ours, _ = tw.event_table(recs, "density", etype, stage=["NREM2", "NREM3"],
                                 freq_lower=lo, freq_upper=hi)
        ours = ours.drop(columns="session").set_index("Subject")
        assert "E5" in ours and (ours["E5"] == 0).all()          # ran, found nothing: zero
        assert "E7" not in ours and "E8" not in ours             # other method / failed run
        pd.testing.assert_frame_equal(ours, bridge, check_names=False, check_exact=True)


SW = [("E1", 1.0, 80.0), ("E1", 1.1, 70.0), ("E2", 1.2, 75.5)]


def test_density_divides_by_the_stages_the_run_searched(tmp_path):
    """Case A: slow waves detected in NREM3 only, spindles in NREM2+NREM3, so analysed_time
    holds both stages. Asked for NREM2 and NREM3, slow-wave density is divided by the 10 min
    of NREM3 the slow-wave run searched (2 events / 10 min = 0.2), not by 40 min."""
    density = pytest.importorskip("turtlewave_hdEEG.density")
    paths = []
    for name in ("sub-01", "sub-02"):
        path = make_runs(tmp_path / name / "wonambi", [
            ("slow_wave", "NREM3", 0.5, 1.25, SW), ("spindle", "NREM2NREM3", 12, 15, EVENTS)])
        add_turtlewave_tables(path, name, {"NREM2": 1800.0, "NREM3": 600.0}, [
            *[(ch, "slow_wave", "m", 0.5, 1.25, "NREM3", 1) for ch in ("E1", "E2", "E5")],
            *[(ch, "spindle", "m", 12, 15, "NREM2NREM3", 1) for ch in ("E1", "E2", "E10")]])
        paths.append(path)
    recs = tw.find_recordings(paths)
    with pytest.warns(UserWarning, match="requested over NREM2\\+NREM3.*searched only NREM3"):
        dens, manifest = tw.event_table(recs, "density", "slow_wave", stage=["NREM2", "NREM3"])
    assert dens.loc[0, "E1"] == pytest.approx(0.2) and dens.loc[0, "E5"] == 0.0
    assert list(manifest["stages"]) == ["NREM3", "NREM3"]
    direct = density.event_density(str(paths[0]), event_type="slow_wave", method="m",
                                   stage="NREM3", freq_lower=0.5, freq_upper=1.25)
    assert dens.iloc[0, 2:].to_dict() == dict(zip(direct["channel"], direct["density_per_min"]))
    spin, manifest = tw.event_table(recs, "density", "spindle", stage=["NREM2", "NREM3"])
    assert spin.loc[0, "E1"] == pytest.approx(2 / 40)        # the spindle run searched both
    assert list(manifest["stages"]) == ["NREM2+NREM3"] * 2


def test_density_over_different_stage_sets_is_refused(tmp_path, monkeypatch):
    """Case B: sub-01 detected jointly over NREM2+NREM3, sub-02 per stage over NREM1, NREM2 and
    NREM3. Asked for all three, their densities would cover different stages: refused, naming
    the subjects per stage set, before TurtleWave is needed."""
    joint = make_runs(tmp_path / "sub-01" / "wonambi", [("spindle", "NREM2NREM3", 12, 15, EVENTS)])
    add_turtlewave_tables(joint, "sub-01", {"NREM1": 60.0, "NREM2": 1800.0, "NREM3": 600.0},
                          [("E1", "spindle", "m", 12, 15, "NREM2NREM3", 1)])
    split = make_runs(tmp_path / "sub-02" / "wonambi", [
        ("spindle", st, 12, 15, EVENTS[i:i + 1]) for i, st in enumerate(("NREM1", "NREM2", "NREM3"))])
    add_turtlewave_tables(split, "sub-02", {"NREM1": 60.0, "NREM2": 1800.0, "NREM3": 600.0},
                          [("E1", "spindle", "m", 12, 15, st, 1)
                           for st in ("NREM1", "NREM2", "NREM3")])
    monkeypatch.setitem(sys.modules, "turtlewave_hdEEG", None)     # importing it now fails
    with pytest.raises(ValueError, match="NREM2\\+NREM3: sub-01; NREM1\\+NREM2\\+NREM3: sub-02"):
        tw.event_table(tw.find_recordings([joint, split]), "density", "spindle",
                       stage=["NREM1", "NREM2", "NREM3"])


def test_density_of_a_run_that_searched_every_requested_stage_is_unchanged(tmp_path):
    pytest.importorskip("turtlewave_hdEEG.density")
    path = make_runs(tmp_path / "sub-01" / "wonambi", [("spindle", "NREM2NREM3", 12, 15, EVENTS)])
    add_turtlewave_tables(path, "sub-01", {"NREM2": 1800.0, "NREM3": 600.0},
                          [(ch, "spindle", "m", 12, 15, "NREM2NREM3", 1) for ch in ("E1", "E2")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")                               # no stage warning
        dens, manifest = tw.event_table(tw.find_recordings([path]), "density", "spindle",
                                        stage=["NREM2", "NREM3"])
    assert dens.loc[0, "E1"] == pytest.approx(2 / 40) and dens.loc[0, "E2"] == pytest.approx(1 / 40)
    assert manifest.loc[0, "stages"] == "NREM2+NREM3"


def test_density_for_stages_the_run_never_searched_is_refused(tmp_path, monkeypatch):
    path = make_runs(tmp_path / "sub-01" / "wonambi", [("slow_wave", "NREM3", 0.5, 1.25, SW)])
    add_turtlewave_tables(path, "sub-01", {"NREM2": 1800.0, "NREM3": 600.0},
                          [("E1", "slow_wave", "m", 0.5, 1.25, "NREM3", 1)])
    monkeypatch.setitem(sys.modules, "turtlewave_hdEEG", None)
    with pytest.raises(ValueError, match="searched NREM3, none of the requested stages"):
        tw.event_table(tw.find_recordings([path]), "density", "slow_wave", stage="NREM2")
