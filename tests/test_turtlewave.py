"""TurtleWave database reader, on small databases built here with the columns the reader uses."""

import json
import sqlite3

import numpy as np
import pytest

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
    make_db(tmp_path / "s01" / "wonambi", EVENTS)
    recs = tw.find_recordings(tmp_path)
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
