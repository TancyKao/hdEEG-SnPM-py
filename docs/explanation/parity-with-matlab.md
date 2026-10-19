# Agreement with the MATLAB toolbox

The MATLAB toolbox hdEEG-SnPM is the reference for this package. This page says what "agrees
with MATLAB" means here, how it is checked, where the two differ on purpose, and what has been
checked without reference to MATLAB.

## How agreement is checked

A permutation test depends on which permutations are drawn. MATLAB and numpy use different
random generators, so the same seed gives different permutations and different p-values. The
check therefore does not share a seed. It shares the permutations.

1. In MATLAB, `scripts/export_parity_fixtures.m` runs each analysis with a recording hook
   switched on. The hook copies out every permutation index the analysis draws. Results and
   indices are saved together as a reference file.
2. In Python, each analysis is run from the same input files and handed MATLAB's indices in
   place of drawing its own.
3. The two sets of results are compared.

What must match:

| Quantity | Requirement |
|---|---|
| Channels tested and channels excluded | identical |
| Matrices handed to the engines after matching, selection, transform, masking and covariate removal | relative difference below 1e-10 |
| Observed statistic at every channel | relative difference below 1e-10 |
| TFCE map; sorted null distributions (maximum statistic, maximum TFCE, cluster size or mass) | relative difference below 1e-9 |
| Corrected p-values | identical |
| Clusters | same channels, same statistic, same p-value |
| Whole-head test | relative difference below 1e-10 (statistic), 1e-8 (p) |
| Excel workbook | same numbers, channel sets and exclusion reasons; layout differs as listed below |

Both programs count a null value as reaching the observed value when the two agree to one part
in 10^12, so a null value that equals the observed value up to rounding is counted the same way
in both, and corrected p-values are identical at every channel of every reference run. (The test
would accept a difference of one count only where the observed value lies within 1e-9 of a null
value; none occurs.)

## What has been checked (18 October 2026)

39 reference runs, checked by 73 tests, all passing:

- 31 t-test and correlation runs: paired, unpaired and one-sample t-tests (both tails, and left
  and right for paired and unpaired); Pearson and Spearman correlation, channel against channel
  and against one behavioural score, with and without covariates; small samples where every
  relabelling is enumerated; data with missing cells, a channel missing in one group, and a
  subject with a missing covariate; ties in Spearman data; `logscale` and `normalize`.
- 8 group-design runs covering 37 permutation loops: `anova1`, `ancova` with two and three
  groups, `regression`, `rmanova`, and `mixed2way` 2 x 2 and 2 x 3 with every post-hoc contrast
  and simple effect.
- 9 Excel workbooks written by MATLAB (paired and unpaired t-test, Pearson correlation with
  covariates, Spearman correlation against one score, two runs with missing cells, `anova1`,
  `rmanova`, `mixed2way` 2 x 2) compared cell by cell with the workbook this package writes.

Most runs use the EGI 256 net. The Compumedics 257 net is checked on four analyses only:
`anova1`, `correlationP`, `pairedT` and `unpairedT`. This was accepted on 18 October 2026
because the engines never see the recording system: it decides which columns are read and which
channels are neighbours, and that step is the same for every analysis.

The TFCE function alone is also compared with 206 stored MATLAB maps and agrees to about 1e-15.

Run it with `pytest -m parity`. The reference files are built on the machine that has MATLAB
and are not in the repository, so these tests are skipped elsewhere.

### Mixed linear model fit (19 October 2026)

The mixed model cannot be compared to rounding. MATLAB's `fitlme` stops its optimiser when the
gradient falls below about 1e-6, so its estimates carry an error of that order, while this
package finds the maximum-likelihood optimum in closed form (see the differences below). The
fit is therefore compared with tolerances, and the correction, which involves no optimiser, is
compared exactly by handing MATLAB's recorded permuted maps to this package's TFCE and cluster
code.

| Quantity | Requirement |
|---|---|
| Channels tested and channels excluded, with their reasons; rows dropped for a missing model variable; channels with no statistic under the backstop (observed and refitted permuted maps) | identical |
| Within-subject share of variation per channel | relative difference below 1e-9 (sweep) |
| Statistic (t or F), Wald statistic, coefficients, standard errors, subject and residual SD | difference below 1e-6 of the largest magnitude in the map |
| Model p-value at every channel | relative difference below 1e-5 |
| Log-likelihood | absolute difference below 1e-8 |
| Permuted maps refitted under MATLAB's indices (every draw of both loops for `lmm_degenerate` and `lmm_factor`, the first 40 draws of each loop elsewhere) | difference below 1e-6 of the largest magnitude in the map, except at channels where MATLAB's optimiser stopped short of the maximum, which must show a higher log-likelihood in this package and are listed by draw |
| TFCE map, sorted TFCE null, sorted cluster-mass null (from MATLAB's permuted maps) | relative difference below 1e-9 |
| TFCE-corrected p-values, clusters and their p-values | identical |
| Whole-head test | relative difference below 1e-6 (statistic), 1e-5 (p) |

Measured worst differences on the reference runs are smaller than these limits: 3.6e-7 on the
statistic, 1.7e-6 relative on the model p, 2.2e-7 on the subject SD and 2e-14 on the
log-likelihood.

Permuted refits agree within the fit tolerance except where MATLAB's `fitlme` stops short of the
likelihood maximum. Refitting all 200 draws of both loops of all six runs, this happens in 9 of
the 400 permuted maps of `lmm_degenerate` (TFCE draws 66, 93, 142, 161, 181, 188; cluster draws
128, 154, 188, counted from 0) and nowhere else; the worst difference is 1.6e-4 of the map
maximum, and at every such channel this package's log-likelihood is the higher one. Two are fits
left at a subject SD of zero whose likelihood gain (9e-8 and 3e-8) is below the 1e-6 that MATLAB's
refit guard asks for; the other seven are interior points where `fitlme` stopped with a
log-likelihood 2e-6 to 1.2e-4 below the maximum. The test records these draws
(`KNOWN_MATLAB_OPTIMISER_SHORTFALLS`) so that a new one fails. Their effect on inference is
negligible: they change a few null maps by at most 1.6e-4 of their maximum, and the TFCE and
cluster corrections computed from MATLAB's own recorded maps are reproduced exactly.

Six reference runs, each with 200 permutations per loop on the EGI net, and one backstop sweep,
checked by 26 tests, all passing on 20 October 2026 (`pytest tests/test_parity_lmm.py`):

- `sleepDepth ~ POWER + (1|Subject)`, continuous, `within_subject`;
- the same with missing cells in four channels, which are excluded;
- `sleepDepth ~ POWER + group + (1|Subject)`, continuous, `within_subject`;
- the same with 33 of 300 rows missing a model variable (group or the response), dropped up
  front;
- `POWER ~ group + (1|Subject)`, factor, `group_label`;
- `lmm_degenerate`: the same factor model with three channels made constant within each subject,
  at within-subject shares of 0, 7e-32 and 9e-7, just under the 1e-6 line. Both programs
  exclude the same three channels with the same reason, and the next channel up (share 0.27) is
  tested.

The backstop sweep (`lmm_backstop_sweep`, built by MATLAB's `scripts/export_lmm_backstop_sweep.m`)
fits `y ~ group + (1|Subject)` with 12 subjects of 5 trials to six columns whose method-of-moments
subject/residual SD ratio is set to 100, 500, 999, 1001, 2000 and 10000. Both programs keep the
first three and give the last three no statistic, and the residual sums of squares, the moment
ratio (relative difference below 1e-9; 1e-6 for the sum of squares with the subject intercepts)
and the within-subject shares agree.

### TurtleWave density

Event density from TurtleWave databases is computed by TurtleWave's own code, so the tests that
check it run only where TurtleWave is installed and are skipped in this package's own
environment. They are run by hand before each release, from the repository root with
TurtleWave-hdEEG checked out beside it and its environment created:

    PYTHONPATH=src ../TurtleWave-hdEEG/.venv/bin/python -m pytest -q tests/test_turtlewave.py

All tests in the file must pass with none skipped. The two tests that compare with the MATLAB
repository's `db_to_group_table.py` also need hdEEG-SnPM checked out beside this repository as
`SnPM_2025`.

## Differences on purpose

The list of deliberate differences is kept in one place, the "Deliberate differences from
MATLAB" section of [CLAUDE.md](../../CLAUDE.md) in the repository root. This section explains
each item on it and adds none.

- **Cluster order**: clusters are numbered by p-value, smallest first, not in the order MATLAB
  happens to find them.
- **Default number of permutations** is 5000 for every analysis.
- **The report** is this package's own design and is not compared with MATLAB's.
- **A missing value in a group-design column** (group, condition, subject, predictor or
  covariate of `anova1`, `ancova`, `regression`, `rmanova` and `mixed2way`) stops the run and
  names the column and rows. MATLAB does not check for it and lets the missing value into the
  design matrix. In the correlation analyses both programs drop a subject with a missing
  covariate and warn.
- **Excel layout**: a significant-channel sheet has one row per channel with its index and label
  (MATLAB writes one row of indices); the `excludedChannels` sheet name carries the analysis
  (MATLAB adds an `Analysis` column); post-hoc status is in the `contents` sheet (MATLAB has a
  `posthoc` sheet); the mixed model adds a `mixedModel` sheet with the formula, the permutation
  scheme, the trials and subjects analysed and the whole-head test, which MATLAB's workbook does
  not have (the `droppedRows` sheet is MATLAB's). Numbers, channel sets and exclusion reasons are
  identical.
- **Mixed-model fit at a subject variance of zero**: with one random intercept the likelihood
  depends on one number, the ratio of the subject SD to the residual SD, and this package
  searches it directly, so it always reaches the maximum, including a ratio of exactly zero.
  MATLAB's `fitlme` works on the same ratio, but zero is always a point of zero gradient for it,
  so an iteration that lands there stops even when a positive ratio fits much better. On one
  reference run this happened at 21 channel fits in 7 of 400 permuted maps, with a
  log-likelihood 4 to 8 lower than the optimum and an F about 2.6 times too large. MATLAB now
  refits any channel whose subject SD comes back at the boundary from a second start and keeps
  the better fit (`snpm_lmm_fitlme.m`), so the two programs agree to the tolerances above, apart
  from the few permuted fits where `fitlme` still stops short of the maximum (listed above).
- **TurtleWave reader versus `db_to_group_table.py`**: the MATLAB repository's bridge script and
  this package's reader build the same tables from TurtleWave databases but differ in four
  places. A database older than TurtleWave 4.3 has no density in the reader, while the bridge
  can divide by hypnogram time when asked to. The reader names a subject by the ID stored in the
  database, the bridge by the folder name. The reader writes the channels of the band asked for,
  the bridge every channel seen in any band. A recording in which the band was never detected
  stops the reader and is an empty row in the bridge. The reasons are in
  [Read TurtleWave databases](../how-to/read-turtlewave-databases.md#compared-with-db_to_group_tablepy).
- **Reasons for untested channels in the circular tier**: a channel can be left out of a
  circular analysis for three reasons: fewer than 8 subjects in a group, log(count) collinear
  with group, or a column with missing angles at a channel that uses the per-channel count
  covariate. The workbook, the HTML report and `excluded_reasons` in the `.json` file name the
  reason that applies to each channel, in the words of MATLAB's parity export. MATLAB's own
  report, workbook and results struct give one combined sentence listing all three reasons for
  every excluded channel, so a reader cannot tell which one applied. That sentence is kept word
  for word in the `.json` file as `circular.exclusion_reason`. The set of excluded channels is
  the same in both programs.

Channels with no variance, and non-finite cells such as the log of zero power, were once listed
here. MATLAB now handles both the same way (the channel is excluded with a reason; a non-finite
cell is a missing cell), so they are no longer differences.

## Behaviour copied from MATLAB that you should know about

- For `tail = "left"` the statistic reported is Data 2 minus Data 1. The report says so.
- The cluster-forming threshold for a correlation is chosen from the number of matched subjects
  before any subject with a missing covariate is dropped.
- The cluster-forming threshold of a one-tailed t-test is the two-sided critical t.
- The cluster statistic of the group designs is the sum of F (or t squared) over the cluster
  divided by the number of channels in the cluster.
- `normalize` z-scores across the selected scalp channels in the t-test and correlation
  analyses, and across every data column in the group designs.

## Defects shared with MATLAB, found in review and fixed (14 October 2026)

Agreement with MATLAB cannot detect a fault that both share. An independent review found two,
a follow-up found a third, and all three are fixed in both this package and MATLAB. (Earlier, on
13 October 2026, `ClusterEnhancement.m` was found to return a result of the wrong length for a
map with every value below the first threshold; it was corrected in MATLAB and both now return
zeros.) The
reference files were regenerated in MATLAB with the new method on 14 October 2026 and all 39
runs agree, including every post-hoc contrast and every `mixed2way` effect.

1. **Pairwise post-hoc contrasts.** "a versus b" used to be permuted with the other levels'
   effects still in the residuals. With a = b and a third level raised across the whole head
   the cluster test rejected at 0.17 to 0.59, and at 0.19 to 0.33 when the third level was
   only less variable. Each pair is now its own two-level test on the rows of those two levels.
   This costs some power (about 0.05 between subjects, about 0.15 within subjects) because the
   error term no longer pools the other levels.
2. **`mixed2way` group effect with incomplete subjects.** Subject means were taken over the
   conditions each subject had, which gave a false-positive rate of 0.61 with a condition
   effect present and a third of subjects missing a condition. `mixed2way` now requires every
   subject to have every condition and stops otherwise, naming the subjects; and every effect
   is tested on one channel set, the channels present in every row.
3. **`mixed2way` condition effect.** The model left a real interaction in the permuted
   residuals, so a crossover interaction could hide a condition effect completely. The
   condition effect is now tested in the full model with the interaction as nuisance. With
   unequal group sizes it is the average across groups with equal weight per group, as in a
   standard mixed ANOVA (it reproduces the F of MATLAB `ranova` on a check table).

False-positive rates after the fixes, from the simulation described below (rows `posthoc:*`
and `mixed2way:*` of the results file):

| Case | TFCE | Cluster |
|---|---|---|
| anova1, a = b, third group raised 3 SD everywhere | 0.041 | 0.050 |
| anova1, a = b, third group half as variable | 0.053 | 0.042 |
| ancova, a = b, third group raised | 0.054 | 0.045 |
| ancova, a = b, third group half as variable | 0.042 | 0.042 |
| rmanova, a = b, third condition raised | 0.036 | 0.045 |
| rmanova, a = b, third condition half as variable | 0.060 | 0.047 |
| mixed2way condition, no condition effect, crossover interaction | 0.064 | 0.056 |
| mixed2way group, condition effect present, one group missing cells | 0.044 | 0.053 |
| mixed2way condition, full null | 0.041 | 0.046 |
| mixed2way group, full null | 0.053 | 0.043 |

## False-positive rates, checked without MATLAB

Agreement with MATLAB shows the two do the same thing. It does not show that thing is right.
`tools/fwe_simulation.py` simulates data with no effect and counts how often anything is
declared significant at 0.05. Noise is spatially correlated on the EGI 178 graph; 1000
simulations per analysis, 199 permutations each. For a true rate of 0.05 the 95% range at this
size is 0.036 to 0.064. The run of 16 October 2026 on the current code,

    python tools/fwe_simulation.py --sims 1000 --perms 199 --workers 6

is in [fwe_2026-10-16.csv](fwe_2026-10-16.csv).

| Analysis | TFCE | Cluster |
|---|---|---|
| pairedT | 0.048 | 0.036 |
| pairedT, 20% of cells missing | 0.039 | 0.025 |
| onesampleT | 0.045 | 0.037 |
| unpairedT | 0.045 | 0.041 |
| unpairedT, one group missing cells in half the channels | 0.041 | 0.033 |
| correlationP | 0.043 | 0.038 |
| correlationP, scattered missing cells | 0.048 | 0.051 |
| correlationS | 0.049 | 0.041 |
| anova1 (3 groups) | 0.050 | 0.061 |
| ancova (covariate with a real effect) | 0.045 | 0.030 |
| regression (nuisance correlated with the predictor) | 0.053 | 0.051 |
| rmanova (3 conditions) | 0.052 | 0.048 |
| mixed2way group | 0.053 | 0.043 |
| mixed2way condition | 0.041 | 0.046 |
| mixed2way interaction | 0.060 | 0.041 |

No rate lies above the range; two TFCE rates in the whole file sit on its edges (the
`mixed2way` crossover case at 0.064, the `rmanova` post-hoc shift case at 0.036). Three cluster
rates lie below it: pairedT with missing cells (0.025), unpairedT with missing cells (0.033) and
ancova (0.030). A rate below 0.05 means the test is conservative, not invalid. For the t-tests
and correlations the cluster statistic is cluster size, a small whole number, and ties count
against significance.

### Correlation with covariates (18 October 2026)

Until 18 October 2026 both programs ran a partial correlation by removing the covariates from
both sides once and then permuting the EEG residuals against the fixed residuals of the other
side. A permuted residual is no longer uncorrelated with the covariates, so the null was too
narrow whenever the covariates explain much of the data: with 10 subjects and 4 covariates the
false-positive rate was 0.87 for the max-statistic, 0.50 for TFCE and 0.23 for clusters. Both
programs now re-project each permuted residual off the covariates before correlating
(Freedman-Lane). The test is approximate, not exact, and the report says so. A redundant
covariate (a duplicate, or a sum of others) is counted once: every degrees-of-freedom statement
uses the rank of the covariates, k. The uncorrected per-channel p-value now uses n - 2 - k
degrees of freedom, as a partial correlation should (it used n - 2). The run stops when
n - 2 - k is below 1 and warns when it is below 5. The reference files for the five covariate
runs were regenerated in MATLAB and Python agrees on all of them.

Partial Spearman ranks the data and the covariates and then removes the covariates linearly
from the ranks. It controls a covariate only as far as the covariate's relation to the ranks
is linear, so it does not test independence given the covariates; use `correlationP` or the
`regression` preset when covariate control matters. The Spearman rows below therefore use
covariates unrelated to the data.

The data model is the one the MATLAB rows were produced with. Each simulation draws n subjects
and k covariates, all independent standard normal. EEG noise is spatially correlated: a
standard-normal value per channel plus half of each neighbour's value on the EGI 178 graph. For
Pearson, both sides depend on the covariates and not on each other: the EEG is the covariates
times a random standard-normal loading per channel, doubled, plus 0.7 times one standard-normal
value per subject shared by every channel, plus the noise; the score is the covariates times a
random standard-normal weight, doubled, plus independent standard-normal error. For Spearman,
the EEG is the shared component plus noise, the score is independent standard-normal, the
covariates are unrelated to both, and all three are ranked. In both cases the covariates (with
an intercept) are removed from the EEG and the score and the Freedman-Lane engines are run with
199 permutations. The run is in [fwe_2026-10-18.csv](fwe_2026-10-18.csv).

    python tools/fwe_simulation.py --sims 1000 --perms 199 --only correlationP_cov:20:3 \
        correlationP_cov:40:5 correlationP_cov:12:2 correlationP_cov:10:4 \
        correlationS_cov:20:3 correlationS_cov:40:5

The rule, set when the method was reviewed, is that a row above 0.06 at 1000 simulations is
re-run with at least 4000 new simulations and passes at 0.055 or below. One row met it, the
Pearson max-statistic at 20 subjects and 3 covariates (0.061), and was re-run; two rows below
the range at 1000 were re-run as well. With 4000 simulations the 95% range for a true rate of
0.05 is 0.0432 to 0.0568 (the `accept_low` and `accept_high` columns of the file).

| Analysis (subjects, covariates) | Simulations | Max-statistic | TFCE | Cluster |
|---|---|---|---|---|
| correlationP (20, 3) | 1000 | 0.061 | 0.057 | 0.045 |
| correlationP (20, 3), repeat | 4000 | 0.0437 | 0.0432 | 0.0395 |
| correlationP (40, 5) | 1000 | 0.045 | 0.046 | 0.037 |
| correlationP (12, 2) | 1000 | 0.041 | 0.039 | 0.027 |
| correlationP (12, 2), repeat | 4000 | 0.0525 | 0.0445 | 0.0355 |
| correlationP (10, 4) | 1000 | 0.040 | 0.018 | 0.018 |
| correlationS (20, 3), unrelated covariates | 1000 | 0.046 | 0.049 | 0.037 |
| correlationS (40, 5), unrelated covariates | 1000 | 0.035 | 0.048 | 0.038 |
| correlationS (40, 5), repeat | 4000 | 0.0467 | 0.0452 | 0.0382 |

The max-statistic row that triggered the rule passes on re-run (0.0437). No rate is above the
range. All three re-run cluster rates (0.0395, 0.0355, 0.0382) sit below the 4000-simulation
range, which means the cluster test is conservative, not invalid. Two things push it down. The
p-value of the t-test and correlation tier counts the identity permutation twice, so with 199
permutations the largest rejection rate attainable under the null is 9 / 200 = 0.045, not
0.05. And the cluster statistic here is cluster size, a small whole number, so
tied sizes count as reaching the observed one and lower the rate further. The same pattern
appears without covariates (16 October run: correlationP cluster 0.038, pairedT 0.036) and in
MATLAB (0.035 at 12 subjects and 2 covariates).

The 10-subject, 4-covariate rows were not re-run. Their TFCE and cluster rates (0.018 and 0.018)
are conservative at 4 error degrees of freedom, as the MATLAB rows were (0.022 and 0.019), and
the warning below 5 error degrees of freedom says so.

The case where every relabelling is enumerated (7 subjects, 2 covariates) is not a row of the
tool. It was checked in MATLAB with a pooled run of 8000 simulations: max-statistic rate 0.0551,
inside the band of 0.025 to 0.065 accepted for that case, where a rate slightly above 0.05 is a
known small-sample property of Freedman-Lane. This package's statistic and its full enumeration
are equal to MATLAB's by parity on the covariate and exact-enumeration reference runs, so the
MATLAB rate applies to it.

Not yet simulated: one-sided tails, and samples small enough for every relabelling to be
enumerated in the t-tests and correlations without covariates.

### Circular tier (19 October 2026)

The three circular analyses were run through the same simulation, 1000 simulations each with
199 permutations, on the EGI 178 graph. Each simulation writes synthetic angle, count and
measure tables to a temporary folder and runs the circular pipeline on them, so the count rule,
the estimability gate of at least 8 subjects per group, the phase-convention check and the
channel set are decided exactly as in a real run. This tier has no max-statistic correction, so
only TFCE and cluster-mass rates are reported.

The data model is the one MATLAB's circular test suite uses. Each subject has an event count at
each channel, drawn as a group median times exp(0.8 z), rounded and at least 2, where z is
spatially correlated noise scaled to unit variance (a standard-normal value per channel plus
half of each neighbour's value). The subject's preferred phase at that channel is zero (the
slow-oscillation up-state peak) plus a deviation of 30 degrees times sqrt(mean count / count)
times another independent field of the same kind, wrapped to the circle. Fewer events therefore
give a noisier angle, which is how event count biases a phase comparison in real data. Angles
are full-precision radians, so Watson's U² does not refuse them as whole-degree input. No group
or measure has any effect on the true phase in any row.

- `circ_phase_group`: Hotelling T² on (cos, sin), two groups of 20, median count 40 in both
  groups, with the per-channel log(count) covariate.
- `circ_phase_group`, count confound: as above, but median count 50 in one group and 20 in the
  other, so the angles of the second group are noisier while their true direction is the same.
  The point-biserial correlation of log count with group is about 0.5 per channel. The row
  shows the rate with the covariate.
- `circ_phase_group`, count confound without the covariate: the same data, but the pipeline is
  given a constant count of 40 for every subject and channel, so the covariate is dropped
  everywhere and the confound is left uncorrected. This is a negative control, not a calibration
  check: it shows what the covariate removes.
- `circ_phase_group`, missing angles: in the front half of the channels every subject's count
  is 40, so the covariate has no variance there and is not used, and 25% of angle cells are
  missing at random; those channels are tested with per-channel deletion. The back half keeps
  variable counts, complete angles and the covariate.
- `circ_phase_group_u2`: Watson's U², same data as the first row (U² cannot take the covariate).
- `circ_corrAngLinear`: 30 subjects, angle against an independent standard-normal measure.
- `circ_corrAngLinear`, missing: 20% of subjects (6 of 30) have no measure and are dropped
  before the analysis, and 20% of the remaining angle cells are missing at random.

The run is in [fwe_2026-10-19.csv](fwe_2026-10-19.csv).

    python tools/fwe_simulation.py --sims 1000 --perms 199 --workers 16 --only \
        circ:phase_group circ:phase_group:count_confound circ:phase_group:missing_front \
        circ:phase_group_u2 circ:corrAngLinear circ:corrAngLinear:missing \
        circ:tier1_unsigned_unadjusted circ:phase_group:count_confound_unadjusted

| Analysis | TFCE | Cluster |
|---|---|---|
| circ_phase_group, counts unrelated to group | 0.050 | 0.051 |
| circ_phase_group, count differs by group (covariate on) | 0.047 | 0.045 |
| circ_phase_group, 25% missing angles where the covariate is not used | 0.047 | 0.052 |
| circ_phase_group_u2 | 0.051 | 0.051 |
| circ_corrAngLinear | 0.053 | 0.043 |
| circ_corrAngLinear, 20% missing measure and 20% missing angles | 0.049 | 0.049 |
| circ_phase_group, count differs by group, covariate off (negative control) | 0.378 | 0.106 |

Every rate except the covariate-off control lies inside the range of 0.036 to 0.064, and none
of those is above 0.06, so no row needed
the 4000-simulation repeat. Unlike the t-test and correlation tier, this tier counts the
identity permutation once, so a rate near 0.05 is attainable, and the cluster statistic is
mass, not size, so ties are rare; no row sits below the range.

The file also holds one negative control, `circ:tier1_unsigned_unadjusted`. It takes the count
confound data, turns each angle into its unsigned distance from zero (the Tier-1 linearised
measure that the pipeline refuses to analyse without a count covariate) and compares the groups
with an unadjusted unpaired t-test. The rates are 0.774 for the max-statistic, 0.999 for TFCE
and 0.992 for clusters: the noisier group has larger distances everywhere, which is why the
refusal exists. MATLAB's figure of 0.280 for this route comes from a design that is not
recorded with it, so it is not compared; this row is a demonstration that the refusal is needed,
not a calibration check.

### Mixed linear model false-positive rates (19 October 2026)

The mixed model was run through the same simulation on the EGI 178 graph, calling the TFCE and
cluster-mass engines as the pipeline does, each with its own permutations. Every refit is a full
mixed-model fit at every channel, so the run is smaller than the others: 600 simulations per row
with 99 permutations per loop, 21 minutes on 16 workers. For a true rate of 0.05 the 95% range
at 600 simulations is 0.033 to 0.067. There is no max-statistic correction in this tier.

Each simulation has 18 subjects with 8 trials each, a random subject intercept whose standard
deviation is 0.2 times the residual standard deviation in both power and the response, and
spatially correlated noise with unit variance at every channel. The subject variance is small on
purpose: about a quarter of the channel fits of the continuous row and almost half of those of
the factor row put the subject variance at exactly zero, so the rows check the boundary where
MATLAB's `fitlme` used to stall. (At 0.4 times the residual standard deviation only 5 to 9% of
fits reach it.) Nothing has an effect.

- Continuous: `DV ~ POWER + (1|Subject)`, the t of POWER, with the response shuffled within
  each subject (`within_subject`).
- Factor: `POWER ~ group + (1|Subject)` with three groups of 6 subjects, the F of group, with
  whole subjects relabelled (`group_label`).

The run is in [fwe_2026-10-19_lmm.csv](fwe_2026-10-19_lmm.csv).

    python tools/fwe_simulation.py --sims 600 --perms 99 --workers 16 \
        --only lmm:continuous lmm:factor

| Analysis | TFCE | Cluster |
|---|---|---|
| mixedmodel, continuous effect, `within_subject` | 0.042 | 0.052 |
| mixedmodel, factor effect (3 groups), `group_label` | 0.037 | 0.048 |

All four rates lie inside the range. The factor TFCE rate (0.037) is near its lower edge, which
means slightly conservative, not invalid.

The run predates the exclusion of channels constant within each subject and the backstop at every
fit (20 October 2026). Neither changes these rows: on 10 of the simulated data sets, with the
observed labelling and 20 permutations each, the smallest within-subject share was 0.66 (the
exclusion line is 1e-6), the largest moment subject/residual SD ratio was 0.64 (the backstop line
is 1000), and no fit was flagged.
