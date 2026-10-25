"""Build synthetic TurtleWave databases for examples/turtlewave_unpaired.toml.

Writes example_output/turtlewave_study/ (git-ignored): one neural_events.db per subject, in the
<subject>/wonambi/ layout TurtleWave uses, and subjects.csv with each subject's group. Sixteen
made-up subjects, s01 to s16, in groups A and B; spindles are detected on the 178 analysed EGI
channels, and group B's spindles are 8 microvolts larger on E129 and its neighbours. Nothing
here comes from a real recording.

    python examples/make_turtlewave_example.py
    snpm run examples/turtlewave_unpaired.toml
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from hdeeg_snpm.montage import get_montage

OUT = Path(__file__).resolve().parent.parent / "example_output" / "turtlewave_study"
PLANTED = ["E129", "E130", "E131", "E118", "E119", "E128", "E142", "E143"]


def main():
    rng = np.random.default_rng(20261025)
    channels = get_montage("egi").labels
    subjects = [f"s{i:02d}" for i in range(1, 17)]
    groups = ["A", "B"] * 8
    for subject, group in zip(subjects, groups):
        rows = []
        for ch in channels:
            shift = 8.0 if group == "B" and ch in PLANTED else 0.0
            for _ in range(int(rng.integers(3, 12))):
                rows.append((ch, float(rng.uniform(0.5, 2.0)), float(rng.normal(40, 6) + shift)))
        folder = OUT / "databases" / subject / "wonambi"
        folder.mkdir(parents=True, exist_ok=True)
        db = folder / "neural_events.db"
        db.unlink(missing_ok=True)
        with sqlite3.connect(db) as con:
            con.execute("CREATE TABLE events (channel TEXT, event_type TEXT, method TEXT, "
                        "stage TEXT, freq_lower REAL, freq_upper REAL, duration REAL, "
                        "peak2peak_amp REAL)")
            con.executemany("INSERT INTO events VALUES (?, 'spindle', 'Moelle2011', "
                            "'NREM2NREM3', 12, 15, ?, ?)", rows)
    pd.DataFrame({"Subject": subjects, "group": groups}).to_csv(OUT / "subjects.csv", index=False)
    print(f"wrote {len(subjects)} databases and subjects.csv under {OUT}")


if __name__ == "__main__":
    main()
