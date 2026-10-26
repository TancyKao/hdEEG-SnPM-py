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

## Run an analysis straight from the databases

Instead of writing the tables first, put a `[turtlewave]` section in the configuration file
of `snpm run`. One command then reads the databases, builds the table of one measure, and runs
the analysis on it:

```toml
comparison   = "unpairedT"
group_col    = "group"
groups       = ["A", "B"]          # Data 1 is group A, Data 2 is group B
permutations = 5000
output_path  = "results"

[turtlewave]
folder        = "/data/study"      # searched for neural_events.db
subjects_file = "subjects.csv"     # Subject plus groups, covariates or scores
event_type    = "spindle"
method        = "Moelle2011"
stage         = ["NREM2", "NREM3"]
freq_lower    = 12
freq_upper    = 15
measure       = "density"          # density, amplitude, duration or count
subject_from  = "auto"
# reject_types = "Artefact"        # density only; by default the set detection stored
```

    snpm run study.toml

Relative paths in `folder` and `subjects_file` are taken from the configuration file's folder.
The section cannot be combined with `data_file`, `data1_file` or `data2_file`, and it fills
`meta_cols` itself. Every table it builds is kept, together with `recordings.csv`, in a
folder named like the run's results with `_tables` added
(`<output_path>/eventStat_<measure>_<event type>_<comparison>_<date>_<time>_tables/`), so each
run keeps its own tables and can be checked and repeated by hand. The summary file and the
report get a "Source" record: the folder, the number of databases analysed and found, the event
type, the method and band of the detection runs selected, the stages and measure, the stages
density was divided by, the reject set its time was cleaned of ("Artefact (as detected)") and
the TurtleWave version that computed it, the interpolated channels
set to missing in the analysed recordings, and any subject left out.

How the table reaches each analysis:

- `onesampleT`: the table is Data 1.
- `pairedT`: `sessions = ["ses-1", "ses-2"]` names the two sessions, which become Data 1 and
  Data 2, matched by subject. A subject with only one of the two sessions is left out, with a
  warning, and listed in the report.
- `unpairedT`: `subjects_file` is joined and the table is split by the values of `group_col`
  (default `group`) listed in `groups`. Group labels are compared as text, as written in the
  file (`01` is not `1`, and `NA` is a label, not a missing value); subjects in any other
  group, or with an empty group cell, are left out and listed. An empty name in `groups` is
  refused.
- `anova1`, `ancova`, `regression`: `subjects_file` is joined and its columns, with `Subject`
  and `session`, become the metadata columns, so `group_col`, `predictor_col` and
  `covariate_cols` name columns of `subjects_file`. These designs take one recording per subject.
- `rmanova`, `mixed2way`: the session is the condition (`condition_col` defaults to `session`,
  `subject_col` to `Subject`); `mixed2way` takes its groups from `subjects_file`.
- `correlationP`, `correlationS`: the table is Data 1 and the column of `subjects_file` named by
  `behavior_col` is Data 2 (when the file has a single score column it is used without naming).
  Covariates of a correlation still come from `covariate_file` with `use_covariates = true`,
  not from `subjects_file`; `use_covariates` without `covariate_file` is refused.
- `mixedmodel` and the circular analyses are refused: they need trial-level rows or phase
  angles, not one value per channel and recording.

The analyses that take one recording per subject (`onesampleT`, `unpairedT`, the correlations,
`anova1`, `ancova`, `regression`) are refused when the databases come from more than one
session, even if every subject has a single database: with group A recorded at one session and
group B at another, session would be mixed into the group comparison. Name the session to keep,
for example `sessions = ["ses-2"]`. Subjects are matched between the databases and
`subjects_file` as text, ignoring a leading `sub-` and letter case; numbers are not padded, so
a database `01` matches a `subjects_file` row `01` but not `1` (the subject column of
`subjects_file` is read as text, so `01` stays `01`). Databases whose subject IDs read as the same
number, such as `01` and `1` or `007` and `7`, are refused as ambiguous; rename the folders. A subject that has a
database but no row in `subjects_file`, or a row but no database, stops the run with both lists;
nobody is dropped silently.

`examples/turtlewave_unpaired.toml` runs on synthetic databases built by
`python examples/make_turtlewave_example.py`. The same settings can be filled in a window
instead of a file: [Use the window (GUI)](use-the-gui.md).

## Which subject a database belongs to

Newer databases store the subject ID; older ones do not. For those the subject is the subject
folder and the session is the `ses...` folder inside it:

    <subject>/<ses-1>/wonambi/neural_events.db    subject and session from the folders
    <subject>/wonambi/neural_events.db            subject from the folder, no session

When a database stores an ID and the folder also names one, they must agree (a leading `sub-`
is ignored). An ID that is the folder subject followed by the session also agrees:
`sub-01_BL` in `sub-01/ses-BL/` (the session with or without `ses-`, letter case ignored). The
subject is then the folder's, `sub-01`, so the sessions of one subject pair up. Any other
disagreement stops the run and names the file. Pass
`subject_from="database"` or `subject_from="folder"` to say which is right, for example when
the folder above the database is not a subject folder.

Two databases resolving to the same subject and session are refused.

## Sessions

Each recording is one row, with a `session` column. To compare sessions within subjects, split
the table by session into two files for `pairedT`, or keep it as one file and use `rmanova`
with `condition_col = "session"`. A `[turtlewave]` section does either for you (above).

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

### Which artefacts density's time excludes

TurtleWave stores the analysed time of each stage once per reject set, the annotation types
whose seconds were removed before detection (artefacts, arousals, movement). Density is divided
by the time stored for the set the detection used, which is what `analysed_time` holds: a
database written by TurtleWave 4.3 records it as two flags, artefacts rejected and arousals
not, read as the set `Artefact`; later versions name the set (`Arousal,Artefact,Move`). The
reader passes that set to TurtleWave explicitly, so it is never replaced by TurtleWave's
current default, which a 4.3 database does not store and which would leave it without a
density. The set is listed per recording in the `reject_types` column of `recordings.csv`.

A database that stores several sets is refused unless `reject_types` in the `[turtlewave]`
section names one (as text, `"Artefact"` or `"Artefact,Arousal"`; `""` or `"none"` is the set
that rejects nothing, written `none` in `recordings.csv`; order and spaces do not matter). A
stored set not in TurtleWave's own spelling (names sorted, comma-joined, no spaces) was written
outside TurtleWave and is refused, naming it. In the window, a study whose databases store
several sets shows a choice of the sets every database stores, which must be made before Run. A named set that a database does not store is
refused with the sets it does store. Databases of one table that use different sets are
refused, with the subjects under each set: their densities are divided by time cleaned in
different ways and are not comparable. `reject_types` is used for density only.

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
identical, and so is density when the database stores analysed time. Both divide density by
the reject set the database stores, by the same rules: several sets need one named
(`--reject-types` in the bridge, `reject_types` here), a table mixing sets is refused, a stored
set not in TurtleWave's key spelling is refused, and the empty set is written `none`. The
differences that
remain are listed, as the authority, in the "Deliberate differences from MATLAB" section of
[CLAUDE.md](../../CLAUDE.md); this section explains them:

- Older databases (before TurtleWave 4.3, no analysed time): the reader has no density; the
  bridge refuses them too, unless `--allow-hypnogram-denominator` is given, in which case it
  divides by hypnogram time of the stages the selected events are stored under and warns that
  this is biased by each recording's artefact load.
- Subject labels: the bridge uses the folder name; the reader uses the ID stored in the database
  when there is one (`sub-` prefix included), except an ID that is the folder subject plus the
  session, where it uses the folder name too.
- Columns: the bridge writes every channel seen in any band; the reader writes the channels of
  the band asked for.
- A recording in which the band was never detected (no event and no record that it was
  processed): the reader stops with an error; the bridge writes an empty row for that recording
  and warns.

## Checking density

Density tests need TurtleWave installed and are skipped without it. Run them in TurtleWave's
environment from the repository root, with TurtleWave-hdEEG checked out beside this repository:

    PYTHONPATH=src ../TurtleWave-hdEEG/.venv/bin/python -m pytest -q tests/test_turtlewave.py
