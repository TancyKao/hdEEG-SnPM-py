# Read TurtleWave databases

TurtleWave-hdEEG writes one `neural_events.db` per recording. This turns a folder of them into
tables of subjects by channels that the analyses accept.

```python
from hdeeg_snpm.io import turtlewave as tw

recordings = tw.find_recordings("/data/study")       # every neural_events.db below this folder
print(tw.available(recordings))                      # event types, methods, stages, bands held

paths = tw.write_tables("/data/study", "tables", "spindle",
                        measures=("density", "amplitude", "duration"),
                        method="Moelle2011", stage=["NREM2", "NREM3"],
                        freq_lower=12, freq_upper=15)
```

This writes `eventStat_<measure>_spindle.csv` for each measure and `recordings.csv`.
**Open `recordings.csv` and check it**: it lists, for every database, the subject and session
it was assigned and where that came from.

## Which subject a database belongs to

Newer databases store the subject ID; older ones do not. For those the subject is the subject
folder and the session is the `ses...` folder inside it:

    <subject>/<ses-1>/wonambi/neural_events.db    subject and session from the folders
    <subject>/wonambi/neural_events.db            subject from the folder, no session

When a database stores an ID and the folder also names one, they must agree (a leading `sub-`
is ignored). If they do not, the run stops and names the file. Pass
`subject_from="database"` or `subject_from="folder"` to say which is right, for example when
the folder above the database is not a subject folder.

Two databases resolving to the same subject and session are refused.

## Sessions

Each recording is one row, with a `session` column. To compare sessions within subjects, split
the table by session into two files for `pairedT`, or keep it as one file and use `rmanova`
with `condition_col = "session"`.

## Measures

- `amplitude`: mean peak-to-peak amplitude per channel. `duration`: mean duration in seconds.
  A channel with no event is empty.
- `count`: number of events; a channel with no event is 0.
- `density`: events per minute, computed by TurtleWave's own `event_density`, which divides by
  the artefact-free time actually searched. It needs TurtleWave installed
  (`pip install hdeeg-snpm[turtlewave]`) and a database that stores that time (TurtleWave 4.3
  or later). A channel the selected run processed without finding an event has density 0;
  channels that only another method processed are not added. When no event matches the request
  and the database lists more than one method for the band, name the method.

### Which stages density is divided by

Density is divided by the analysed time of the requested stages that the selected run actually
searched, not of every requested stage. The stages a run searched are read as TurtleWave reads
them: from its processing record for that event type, method and band, else from the stages
listed for the detection run, else from the stages its events are stored under. For example,
with slow waves detected in NREM3 only and spindles in NREM2 and NREM3, a slow-wave request for
`stage=["NREM2", "NREM3"]` divides by NREM3 time alone; dividing by NREM2 + NREM3 time would
scale each subject's density by their share of NREM3, which differs between groups.

When the runs searched fewer stages than requested, a warning says so, and the
`stages` column of `recordings.csv` gives the stages used for each subject. Within one
table every subject must use the same stages: if one subject's run searched NREM2 and NREM3
jointly and another's searched NREM1, NREM2 and NREM3 separately, a request for all three is
refused and the error lists the subjects under each stage set. Request only the stages every
run searched, or redetect so the runs match.

Name the method, stage and band whenever a database holds more than one; a request that matches
two is refused, not pooled. A stage request is read the way TurtleWave reads it for density:
`stage=["NREM2", "NREM3"]` (or `"NREM2NREM3"`) covers a run that searched both stages jointly
and runs that stored NREM2 and NREM3 separately, which are then pooled. `stage="NREM2"` does not
cover a joint run, because its events cannot be split by stage. Two runs that both cover a stage
(for example a per-stage rerun of the same night) are refused rather than added together, for
density as for the other measures.

## Interpolated channels

Databases from TurtleWave 4.5 record which channels were interpolated. Those channels are set
to missing for that recording and listed in `recordings.csv`. Pass `drop_interpolated=False`
to keep them. Older databases do not record this, so nothing can be dropped for them.

## Compared with `db_to_group_table.py`

The MATLAB repository's bridge script writes the same `eventStat_<measure>_<band>.csv` layout
and follows the same rules for selecting a run, pooling stage tokens, refusing overlapping
runs, zero-event channels and interpolated channels. Both refuse density for a band with no
event in scope whose processing record lists more than one method, because which method's
channels count as zero is then undefined. On the same databases amplitude and duration are
identical, and so is density when the database stores analysed time. The differences that
remain are listed, as the authority, in the "Deliberate differences from MATLAB" section of
[CLAUDE.md](../../CLAUDE.md); this section explains them:

- Older databases (before TurtleWave 4.3, no analysed time): the reader has no density; the
  bridge refuses them too, unless `--allow-hypnogram-denominator` is given, in which case it
  divides by hypnogram time of the stages the selected events are stored under and warns that
  this is biased by each recording's artefact load.
- Subject labels: the bridge uses the folder name; the reader uses the ID stored in the database
  when there is one (`sub-` prefix included).
- Columns: the bridge writes every channel seen in any band; the reader writes the channels of
  the band asked for.
- A recording in which the band was never detected (no event and no record that it was
  processed): the reader stops with an error; the bridge writes an empty row for that recording
  and warns.

## Checking density

Density tests need TurtleWave installed and are skipped without it. Run them in TurtleWave's
environment from the repository root, with TurtleWave-hdEEG checked out beside this repository:

    PYTHONPATH=src ../TurtleWave-hdEEG/.venv/bin/python -m pytest -q tests/test_turtlewave.py
