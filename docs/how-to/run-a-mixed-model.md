# Run a mixed linear model

Use this when the data are **trial-level**: several rows per subject (awakenings, epochs, nights,
trials), and the question is how a per-trial measure relates to channel power, or whether power
differs between groups once every trial is kept. At each channel a linear mixed model with a
random intercept per subject is fitted, the effect of interest is mapped as a t or an F, and the
map is corrected across channels by permutation with threshold-free cluster enhancement (TFCE)
and a cluster-mass test. This follows Stephan et al. (2021).

If each subject has one value per channel and condition, this is the wrong page: use the t-tests
or group designs of [Analyse a table of subjects by channels](analyse-a-table.md), which are
faster and exact.

## Prepare the file

One file, **long format**: one row per trial. It holds:

- a subject column (`Subject` unless you say otherwise in `lmm_subject`);
- the metadata the model needs: the response if it is not power, any fixed-effect variables, and
  a group column if the effect is between groups;
- one column per channel, named as the recording system names them (`E1`, `E2`, ...).

Every column you list in `lmm_meta_cols` is metadata; every other column is taken as a channel.
The channels kept must match the recording system's montage, as in the other analyses.

`examples/data/lmm_long.csv` is a synthetic example: 20 subjects with 15 trials each (300 rows),
two groups (`G1`, `G2`), a trial number `time`, a response `sleepDepth`, and power at the EGI
channels, with a power-to-response relation planted around E129.

## Write the configuration

`examples/lmm.toml` asks whether power predicts sleep depth, within subjects:

```toml
comparison      = "mixedmodel"
channels        = "egi"
datatype        = "absolute"
data_file       = "data/lmm_long.csv"
lmm_meta_cols   = ["Subject", "group", "time", "sleepDepth"]
lmm_dv          = "sleepDepth"
lmm_fixed       = "POWER"
lmm_random      = "(1|Subject)"
lmm_effect      = "POWER"
lmm_effect_type = "continuous"
permutations    = 500
seed            = 20261013
output_path     = "../example_output"
```

    snpm run examples/lmm.toml

The model fitted at each channel is `sleepDepth ~ POWER + (1|Subject)`, where `POWER` stands for
that channel's column. The keys of this analysis:

| Key | Values | Default | Meaning |
|---|---|---|---|
| `data_file` (or `data1_file`) | path | required | the long-format file |
| `data_sheet` | sheet name | first sheet | for an Excel file |
| `lmm_meta_cols` | list of column names | required | the columns that are not channels |
| `lmm_dv` | a metadata column, or `POWER` | required | the response. `POWER` makes channel power the response |
| `lmm_fixed` | Wilkinson formula | required | fixed effects, written with `+`, `*` and `:`; `POWER` is the channel. It must contain `POWER` unless `POWER` is the response |
| `lmm_random` | `(1\|<column>)` | required | the random part; only a random intercept for one grouping column is accepted |
| `lmm_effect` | a term of `lmm_fixed` | required | the effect that is mapped and tested |
| `lmm_effect_type` | `continuous`, `factor` | required | `continuous` maps the t of one coefficient; `factor` maps the F of all the coefficients of a term |
| `lmm_perm` | `within_subject`, `group_label` | from `lmm_effect_type` | the permutation scheme (next section). Leave it out |
| `lmm_subject` | column name | `Subject` | the subject column the permutations respect |
| `lmm_group` | column name | `group` | the column relabelled by `group_label` |
| `channels`, `datatype`, `permutations`, `seed`, `tail`, `output_path` | | | as for every analysis ([Parameters](../reference/parameters.md)) |

Text columns, and the subject, group and grouping columns, are treated as categorical with the
first level in sorted order as the reference, as MATLAB's `fitlme` does. A numeric column used as
a fixed effect is treated as a number.

For a difference in power between groups, make power the response and test the group term as a
factor:

```toml
lmm_dv          = "POWER"
lmm_fixed       = "group"
lmm_random      = "(1|Subject)"
lmm_effect      = "group"
lmm_effect_type = "factor"
```

With three or more groups this is one F per channel for any difference among them, which the
two-group t-tests cannot express at trial level.

## Effect type and permutation scheme go together

The permutation must break exactly the link being tested and keep everything else. The scheme
is therefore chosen from the effect type:

| `lmm_effect_type` | Scheme | What is permuted | Use for |
|---|---|---|---|
| `continuous` | `within_subject` | the response is shuffled among each subject's own rows | a trial-level relation, such as power predicting a per-trial score |
| `factor` | `group_label` | whole subjects are relabelled between groups; each subject keeps all its rows | a between-subject effect, such as a group difference |

Shuffling within subjects never moves a subject from one group to another, so it gives no null
for a between-group effect: the observed group difference is reproduced in every permutation
and nothing can be significant. Relabelling groups, for its part, does not touch the response or
power, so for a trial-level relation it leaves the model unchanged and gives no null either.
`lmm_perm` exists to reproduce MATLAB runs; override it only if you know why.

`within_subject` shuffles the response, so the response must be a metadata column (not `POWER`)
and must vary within at least some subjects. `group_label` relabels the group column, so the
effect must contain that column. A setting that breaks either rule would leave every permuted map
equal to the observed one, and is refused before any fit (next section). A subject with only one
row cannot be shuffled and adds nothing to the null; the run warns and names it.

## What is refused

These checks run before any model is fitted. The run stops, and says why, when:

- `tail` is not `both` (`oneSidedTail`): the map is a two-sided t or an F, and clusters are
  formed on the two-sided model p;
- under `group_label`, `lmm_effect` does not contain the group column (`effectNotPermuted`). A
  part of the effect counts as the group column when it is the column itself or a coefficient of
  it (`group_B`, `group_true`, also inside an interaction such as `group_B:POWER`); another
  column whose name merely starts with it, such as `group_size`, does not count;
- under `within_subject`, the response is `POWER` or is not listed in `lmm_meta_cols`
  (`dvNotMetadata`);
- a metadata response is constant within every subject: the share of its variation that lies
  within subjects is at most 1e-6 (`dvConstantWithinSubject`). Under `within_subject` the
  shuffle has nothing to permute; under `group_label` every fit would be a perfect fit. Analyse
  subject means with `anova1` or `unpairedT` instead;
- under `within_subject`, every variable of the effect is constant within each subject
  (`effectConstantWithinSubject`): for example `group`, `group_B` or a subject-level covariate
  such as age. Shuffling the response inside each subject leaves every subject's mean in place,
  so such an effect barely moves under the shuffle and there is no null for it (in simulation
  the false-positive rate reached 0.35 at a nominal 0.05). The check takes the largest
  within-subject share among the effect's variables (a text column through one indicator per
  level) and refuses at 1e-6 or below. An effect containing `POWER`, or any variable that varies
  within subjects (`group:time`), runs. Use `group_label` for a between-group effect;

- the random part is anything but `(1|<column>)`: random slopes such as `(POWER|Subject)`,
  several grouping columns, and crossed effects such as `(1|Subject) + (1|time)` are not
  available in this version;
- the fixed part removes the intercept (`- 1`, `0`), or contains a term that cannot be read;
- `POWER` is both the response and a fixed effect, or neither;
- `lmm_effect` is not a term (factor) or a coefficient (continuous) of `lmm_fixed`
  (`unknownEffect`). The message lists the terms or coefficients as `fitlme` names them: terms
  joined with `:` in table order and without the intercept, coefficients as `<column>_<level>`
  for a categorical column. A factor term may be written in any variable order: `POWER:group`
  is taken as `group:POWER`, and the outputs use that name. Which level is the reference (and so
  which coefficients exist, and the sign of their t) follows `fitlme`: the subject, group and
  grouping columns are converted to categories, sorted (text alphabetically, numbers in numeric
  order, a true/false column as `group_false`, `group_true`); any other text column keeps its
  levels in the order they first appear in the file, the first being the reference (with `pre`
  in the first row the coefficient is `cond_post`); a true/false column that is not converted
  has reference false and coefficient `<column>_1`; a numeric column is a continuous predictor.
  Under `group_label`, a coefficient of a group level that is not in the data (`group_C`) is
  refused as `unknownEffect`, with the real coefficients listed;
- a column the model needs is not in the file;
- fewer than 3 subjects are left once incomplete rows are dropped;
- under `group_label`, a subject's group changes between its rows, or a group has fewer than 2
  subjects (relabelling cannot build a null for a one-subject group);
- the number of channels does not match the recording system, or no channel is left to test
  once incomplete, constant-within-subject and flat channels are removed
  (`noEvaluableChannels`; the message lists the reasons).

## Missing values

**Rows with a missing model variable are dropped first.** A row with an empty cell (or, in a
numeric column, +Inf or -Inf) in the response, a fixed-effect variable, the grouping or subject column, or (under `group_label`) the
group column cannot enter the model at any channel, so it is removed from the metadata and the
channel matrix together before anything else. The run warns with the number of rows dropped,
the count per column, and any subject left with no rows. The record is kept in every output:
the `.json` file (`mixed_model.dropped_rows`: the rows dropped, counted from 0 in the file as
read, the count per model variable, the subjects lost, and the rows and subjects before and
after), the `droppedRows` sheet of the workbook (a summary line, then the count per model
variable, as MATLAB writes it), the report, and the `dropped_rows` array of the `.npz` file. MATLAB drops
the same rows (`snpm_lmm_drop_missing_meta.m`).

**Channels with a missing value are not tested.** After the drop, a channel is tested only if it
has a value in every remaining row. That set is fixed before the first permutation and used for
the observed map and every permuted map; the other channels are listed with the reason "missing
data: channel not complete across all analysed rows". If more than 10% of the montage is lost
this way, the run adds a note: dropping the few trials or subjects that carry the gaps usually
costs less than dropping the channels.

## Channels constant within subjects, and fits with nothing left over

A mixed model separates variation between subjects from variation within them. A channel whose
value does not change across a subject's trials (a flat channel, or one per-subject value copied
onto every trial) has only the first kind. Two rules deal with it, the same as in MATLAB.

**Excluded once, up front.** Before any fit, each complete channel's within-subject share of
variation is computed: the sum of squares around each subject's own mean divided by the sum of
squares around the overall mean. It is grouped by subject only, so it does not depend on any
labelling, and a channel removed by it is absent from the observed map and every permuted map.
A channel with a share of at most 1e-6 is excluded when:

- it is the response (`lmm_dv = "POWER"`), with the reason "channel value constant within each
  subject: no within-subject variation for the mixed model to estimate; analyse subject means
  with anova1/unpairedT";
- it is the predictor under `within_subject`, with the reason "predictor constant within each
  subject: its effect is between-subject only, which the within-subject shuffle cannot test (use
  group_label or regression on subject means)". Left in, such a channel keeps the same t in every
  permuted map and sets the maximum-statistic threshold for the whole head.

Under `group_label` with a metadata response and `POWER` as a predictor, a channel constant
within each subject is kept: relabelling whole subjects moves each subject's response and
channel value together, so the channel acts as a subject-level covariate and its null is sound.
Only a channel with one value on every trial is excluded there, with the reason "same value in
every subject", because it cannot be told apart from the intercept at any labelling. (MATLAB
also excludes subject-constant channels when the random part has a `POWER` slope by subject;
this package fits a random intercept only, so the case does not arise.)

The share of every channel is kept in the `.npz` file as `within_ratio` (NaN for an incomplete
channel), so a channel near the line can be checked.

**Checked at every fit.** A channel that passes can still give a fit with nothing left over at one
particular labelling, observed or permuted. After every fit the data are checked directly: the
fit has no usable residual variation when the fixed effects, or the fixed effects and the subject
intercepts together, reproduce the response to within 1e-20 of its uncentred sum of squares, or
when the ratio of subject SD to residual SD, estimated by the method of moments, is 1000 or more.
Such a fit has no statistic: NaN in the map and 0 towards cluster mass. An observed channel caught
this way is shown as NaN and listed with the reason "perfect fit, statistic undefined". Real
trial-level power has ratios in single digits, so ordinary channels are never affected.

## How the model is fitted

Each channel is fitted by maximum likelihood, as MATLAB's `fitlme` with its defaults. With one
random intercept the likelihood depends on a single number, the ratio of the subject standard
deviation to the residual standard deviation, so the fit is a one-dimensional search done for
all channels at once. It always finds the optimum, including a subject variance of exactly zero
when the data put it there; there is no optimiser that can stop early. The t uses residual
degrees of freedom (rows minus the number of fixed coefficients); the F of a factor is the
marginal Wald test divided by its number of coefficients, as `anova(lme)` reports. A channel
whose fit cannot be made (fewer than 5 rows, a rank-deficient design, or no residual variance
left, as for a flat channel or one whose value is constant within each subject) has no
statistic and adds nothing to a cluster, in the observed map and in every permuted map.

A permuted map is a full refit at every channel, so a mixed-model run is far slower than a
t-test: the example takes about a minute for 500 permutations. Use a few hundred permutations
while setting up and 5000 or more for the result you report.

## Read the output

The four usual files are written (see [Output files](../reference/outputs.md)). What differs
for this analysis:

- **Statistic.** t for a continuous effect, F for a factor. The uncorrected p is the model's own
  p-value (two-sided for t).
- **Correction.** TFCE with the largest enhanced value over channels as the null, and a cluster
  test on channels with model p < 0.05 whose statistic is the cluster **mass**: the sum of the
  Wald statistic (t² or F) over the cluster divided by the number of channels in it, the
  recipe of Stephan et al. (2021). The two corrections draw their own permutations, as in
  MATLAB. The test is two-sided only: `tail` must be `both` (anything else is refused before the
  fit), and clusters are formed on the two-sided model p.
- **Whole-head test.** The same model fitted once to the power averaged over the tested
  channels. It is parametric, not corrected, and descriptive.
- **`.xlsx`** adds two sheets. `droppedRows`, as in MATLAB, records the rows dropped for a
  missing model variable. `mixedModel`, which MATLAB's workbook does not have, gives the
  formula, random structure, effect and effect type, statistic, residual degrees of freedom,
  permutation scheme, trials and subjects analysed, the cluster statistic and the whole-head
  test.
- **`.json`**, under `effects[0].mixed_model`: the same record, plus the names of the fixed
  coefficients and the number of permutations that gave no finite statistic anywhere (each
  such permutation enters the TFCE null as +Inf, which can only make the test more
  conservative).
- **`.npz`**, besides the maps: `evaluable`, `wald`, `beta` (every coefficient), `se`, `loglik`,
  `sd_subject`, `sd_residual`, `boundary` (the subject variance was fitted as zero),
  `tfce_null`, `cluster_mass_null` and `dropped_rows`.

A significant continuous effect means the response changes with power at those channels within
subjects. A significant factor means the group means of power differ there, after the trials of
each subject are pooled through the random intercept.
