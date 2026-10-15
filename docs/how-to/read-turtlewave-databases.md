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
  or later).

Name the method, stage and band whenever a database holds more than one; a request that matches
two is refused, not pooled.

## Interpolated channels

Databases from TurtleWave 4.5 record which channels were interpolated. Those channels are set
to missing for that recording and listed in `recordings.csv`. Pass `drop_interpolated=False`
to keep them. Older databases do not record this, so nothing can be dropped for them.
