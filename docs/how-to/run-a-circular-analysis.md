# Run a circular (phase) analysis

Use this when the quantity per subject and channel is a **preferred phase**: the angle on the
slow oscillation at which a subject's spindles (or other events) tend to occur. Coupling
strength and event density are ordinary numbers and belong in the t-tests or group designs of
[Analyse a table of subjects by channels](analyse-a-table.md); they need none of this.

Why the tests are built this way, and what a result does and does not license, is explained in
the MATLAB toolbox's
[About circular statistics for phase](https://github.com/TancyKao/hdEEG-SnPM/blob/main/docs/explanation/circular-statistics-for-phase.md).

## Pick the analysis

| `comparison` | Question | Statistic per channel | Files |
|---|---|---|---|
| `circ_phase_group` | Do two independent groups differ in preferred phase? Use this one by default. | Hotelling T² on (cos, sin), adjusted for log(event count) at that channel, as an F | two angle files, two count files |
| `circ_phase_group_u2` | Same design, any difference in the phase distribution | Watson's two-sample U²; cannot adjust for event count | two angle files, two count files |
| `circ_corrAngLinear` | Does a per-subject measure vary with phase? | circular-linear F of the measure on [1, cos, sin] | one angle file, one measure file |

All three are between-subject. If both angle files hold the same subjects (two nights, two
conditions), this is the wrong page: there is no within-subject circular test.

## Prepare the files

Every file is wide: a `Subject` column and one column per channel, named as the recording
system names them (`E1`, `E2`, ...). An empty cell is a missing value.

- **Angle files** (`data1_file`, `data2_file` for the group analyses; `data1_file` for
  `circ_corrAngLinear`). Radians on [0, 2π) or (−π, π], or degrees on [−360, 360]. For the two
  group analyses the second angle file must have the same channel columns as the first (order
  does not matter); the subjects of the two groups are different people.
- **Count files** (`count1_file`, `count2_file`), required for both group analyses: the number
  of detected events behind each angle, one file per group, with exactly the same subjects and
  channel columns as that group's angle file (order free). Counts must be whole numbers of at
  least 0. A mismatch in subjects or channels stops the run and names the entries.
- **Measure file** (`data2_file` for `circ_corrAngLinear`): a `Subject` column and the measure.
  If it has more than one other column, name the one to use in `measure_col`. Subjects are
  matched by ID; those present in both files are analysed.

## Write the configuration

`examples/circ_phase_group.toml` runs the primary analysis on synthetic data with a phase shift
planted around E129:

```toml
comparison      = "circ_phase_group"
channels        = "egi"
data1_file      = "data/circ_phase_groupA.csv"
data2_file      = "data/circ_phase_groupB.csv"
count1_file     = "data/circ_phase_countsA.csv"
count2_file     = "data/circ_phase_countsB.csv"
circ_units      = "rad"
circ_convention = "literature_uppeak0"
permutations    = 1000
seed            = 20261013
output_path     = "../example_output"
```

    snpm run examples/circ_phase_group.toml

For `circ_phase_group_u2` change only `comparison`. For the circular-linear test:

```toml
comparison      = "circ_corrAngLinear"
channels        = "egi"
data1_file      = "data/circ_anglinear_angles.csv"
data2_file      = "data/circ_anglinear_measure.csv"
measure_col     = "measure"
circ_units      = "rad"
circ_convention = "literature_uppeak0"
permutations    = 1000
```

The keys particular to this tier:

| Key | Values | Default | Meaning |
|---|---|---|---|
| `circ_units` | `rad`, `deg` | **none, required** | units of the angle files. Values out of range for the declared unit stop the run; nothing is converted silently |
| `circ_convention` | `literature_uppeak0`, `yasa_uppeak0`, `luna_zerocross0`, `turtlewave_pre_v4`, `custom` | **none, required** | where the exporting tool puts zero (next section) |
| `circ_zero_offset_deg` | number | | degrees added so that 0 lands on the up-state peak; only with `custom` |
| `count1_file`, `count2_file` | path | | event counts, required for both group analyses |
| `count1_sheet`, `count2_sheet` | sheet name | first sheet | for Excel count files |
| `measure_col` | column name | the only column | the measure, for `circ_corrAngLinear` |
| `circ_count_covariate_mode` | `per_channel`, `per_subject` | `per_channel` | log count at each channel, or the subject's log mean count at every channel. Keep the default: an event-count imbalance acts channel by channel |
| `circ_collinearity_max` | number | 0.8 | a channel is not tested when \|point-biserial r(log count, group)\| exceeds this |

`tail` and `datatype` are fixed. All three statistics are non-negative with one upper tail, so
`tail` must be `both` (or left out), and the angle cannot be log-transformed or z-scored, so
`datatype` must be `absolute` (or left out). Anything else stops the run.

Units and convention have no default because a wrong guess rotates or scrambles every angle and
no later check can always tell.

## Declare the zero convention

The analysis puts 0 at the slow-oscillation up-state peak. Tell it where your tool put zero:

| `circ_convention` | Zero in the exported angle | Rotation applied |
|---|---|---|
| `literature_uppeak0` | up-state peak | 0° |
| `yasa_uppeak0` | up-state peak (YASA) | 0° |
| `luna_zerocross0` | positive-to-negative zero crossing (Luna `COUPL_ANGLE`) | +90° |
| `turtlewave_pre_v4` | TurtleWave exports before v4.0, which are exactly 180° out | +180° |
| `custom` | anything else | `circ_zero_offset_deg` |

Rotate in the configuration, never by hand in the file: the rotation applied is recorded in the
`.json` file and printed in the report.

**The inversion check.** Before testing, the pooled mean direction of both groups together is
computed over the frontal third of the tested channels. If it lies within 70° of exact
antiphase to the up-state peak (110° to 250°), the run stops with `circPhaseConventionInverted`:
down-state coupling of that size is the signature of a convention error, not a finding. Re-export
from TurtleWave v4.0 or later, or declare `turtlewave_pre_v4`. A deviation between 70° and 110°
only warns. Pooling both groups makes the check blind to group labels, so it cannot bias the
test. Its result is in `.json` under `circular.inversion_check`.

**Angular resolution.** Whole-degree or coarsely rounded angles stop `circ_phase_group_u2`
(`circResolutionTooCoarse`), because U² works on ranks and ties then change the result. The
other two analyses warn and continue. Export the angle at full precision, or use
`circ_phase_group`.

## Missing values and untested channels

- **Counts.** A cell with an angle but no usable count (empty or 0) loses its angle, with a
  warning. If more than 5% of angle cells lack a count, the run stops: that pattern means the
  count files belong to a different detection run.
- **The measure** (`circ_corrAngLinear`). A subject with no measure value cannot enter the test
  at any channel, so it is dropped from the whole analysis before anything else, with a warning;
  the subjects dropped are listed in the notes, the report and the `.json` file. Fewer than 8
  subjects left stops the run.
- **Angles.** A missing angle stays with its subject's row and is left out at that channel only,
  so the row set is the same for the observed map and every permuted map.

A channel is tested only if it has at least 8 subjects with an angle in each group (8 in total
for `circ_corrAngLinear`), its log count is not collinear with group (|point-biserial r| above
`circ_collinearity_max`, or the two groups' counts do not overlap), and, where log count varies
enough to be a covariate, no subject is missing an angle there. The covariate is left out at a
channel where log count barely varies (standard deviation below 0.05); there missing angles are
simply left out. `circ_phase_group_u2` applies the same channel rules, so its channel set
matches the Hotelling analysis on the same files. This channel set is fixed from the observed data before the
first permutation and checked again on the first permuted map; a mismatch stops the run rather
than produce a miscalibrated correction. If no channel survives, or log count is collinear with
group at more than a quarter of the channels, the run stops; the second case can only be
repaired upstream by drawing the same number of events per subject in both groups.

Every channel that is not tested is listed with its reason in the report, the workbook and the
`.json` file.

## The Tier-1 guard on linearised angles

`hdeeg_snpm.circ.linearise` turns angles into an ordinary number per channel so they can go
through the t-tests or group designs, and `hdeeg_snpm.circ.linearise_table` writes that as a
table with a `circ_provenance` column (list it in `meta_cols`). The **signed** measure, the
deviation from the pooled mean direction, can be analysed like any other number. The
**unsigned** measure, |deviation| from a chosen phase, rises with estimation noise, so a group
difference in event count alone produces a false-positive rate of 0.28 at a nominal 0.05. Every
non-circular analysis therefore reads the provenance stamp first and refuses the unsigned
measure (`circUnsignedNeedsCountCovariate`) unless the analysis is `ancova` or `regression` with
the event count among `covariate_cols`. A covariate whose name contains `count` is recognised;
otherwise name it in `circ_count_col`.

## Read the output

The four usual files are written (see [Output files](../reference/outputs.md)). What differs for
this tier:

- **No whole-head test.** An arithmetic mean of angles is meaningless, and a circular mean over
  the head would average across a real front-to-back phase gradient. In its place the report
  has a descriptive panel per group: mean direction, circular standard deviation, resultant
  length R and the Rayleigh test, pooled over the tested channels. It carries no inference.
- **Correction.** TFCE (step 0.1, or 0.005 for U², whose values are small) and a cluster test
  whose statistic is the cluster **mass**: the sum of the statistic over the cluster divided by
  the number of channels. Clusters form from channels with uncorrected parametric p < 0.05.
  FieldTrip's `maxsize`, used in some published phase studies, is a different statistic; the
  report says so.
- **`.json`**, under `effects[0].circular`: the import record (units, convention, rotation
  applied, resolution), the count covariate report (mode, point-biserial median and maximum,
  channels where the covariate was dropped or the channel was collinear, cells without a
  count), the inversion check, group sizes, the subjects analysed, subjects dropped for a
  missing measure, the permutation type, and the descriptive panel.
- **`.npz`**, besides the maps: `evaluable`, `usecov` (the covariate was used), `collinear`,
  `n_finite`, `mean_direction_deg` (one row per group, or one row for `circ_corrAngLinear`),
  and for the group analyses `n_group1`, `n_group2` and `point_biserial`; `rho` for
  `circ_corrAngLinear`.
- **Notes** at the top of the report and in `.json`: the event-count / group association
  (median and largest point-biserial r). Read it before the p-values. Above about 0.5 the
  covariate absorbs most of the group contrast and the test cannot separate phase from event
  rate, whatever the p-value. `circ_phase_group_u2` also warns that its map is not adjusted for
  event count.

A significant cluster in a group analysis means the phase distributions differ there: a shift
in mean direction, a difference in concentration, or both. It does not say which, nor which
group is earlier. For `circ_corrAngLinear` it means the measure varies with phase, not that it
is higher at any particular phase.
