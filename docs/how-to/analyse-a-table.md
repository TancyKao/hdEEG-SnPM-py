# Analyse a table of subjects by channels

Input is a CSV file or an Excel sheet with one row per subject (or per subject and condition)
and one column per channel. Channel columns are matched to the recording system by name, so
extra channels the system does not analyse are ignored and column order does not matter within
one file.

## Two files: t-tests and correlation

`pairedT`, `unpairedT`, `onesampleT`, `correlationP` (Pearson) and `correlationS` (Spearman)
take `data1_file` and `data2_file`. Each file is a `Subject` column plus channel columns.

- `pairedT` and `onesampleT` pair the files row by row. Put the subjects in the same order.
  `onesampleT` with no `data2_file` tests one file against zero.
- `unpairedT` treats the files as two groups.
- Correlation matches subjects by the `Subject` column and analyses those present in both.
  Data 2 may be a `Subject` column plus one behavioural score; the score is then correlated
  with every channel.
- For a correlation, add `use_covariates = true` and `covariate_file` (first column the
  subject ID, then one column per covariate) for a partial correlation. Both sides are
  residualised on the covariates and each permutation re-projects the permuted EEG residuals
  off them before correlating (Freedman-Lane). This test is approximate, not exact. A subject
  with a missing covariate value is dropped from the whole analysis with a warning. The run
  stops when fewer than 1 error degree of freedom is left (n - 2 - covariates), and warns
  below 5. With `correlationS` the covariates are removed only as far as their relation to the
  ranks is linear; use `correlationP` or the `regression` preset when covariate control is the
  point.

The second file is read position by position against the first, so both must list the same
channels in the same order. The run stops with the mismatched positions named if they do not.

## One file: group designs

`anova1`, `ancova`, `regression`, `rmanova` and `mixed2way` take one `data_file` holding the
channels and the design columns. List the design columns in `meta_cols` and say which is which:

| Preset | Needs |
|---|---|
| `anova1` | `group_col` |
| `ancova` | `group_col`, `covariate_cols` |
| `regression` | `predictor_col`, optional `covariate_cols` |
| `rmanova` | `condition_col`, `subject_col` (one row per subject and condition) |
| `mixed2way` | `group_col`, `condition_col`, `subject_col` |

Presets with three or more levels also compare every pair of levels. Each pair is its own
test on the rows of those two levels only, so a third level cannot affect it. `mixed2way` reports the
group effect, the condition effect, the interaction, and the condition effect inside each
group. The between-subject presets need exactly one row per subject and stop if a subject
repeats. `mixed2way` needs every subject to have exactly one row per condition and stops,
naming the subjects, if any does not.

The run also stops when the effect cannot be told apart from the nuisance columns (the
covariates, and the subject columns of a repeated-measures design): a covariate that repeats
the predictor, a covariate equal to twice the predictor plus a constant, a covariate that
splits the subjects exactly as the groups do, or an `rmanova` condition seen only in subjects
with a single row. Its contrast would not be estimable, so any statistic would be arbitrary.
The message names the aliased columns, for example `predictor logAHI is a linear combination
of covariate logAHI` (`effectAliased`). When only the covariates depend on each other (age in
years and in months), the effect is still estimable: the run warns
(`GLMNuisanceRankDeficient`) and gives the same maps as with the redundant covariate removed.
A pair of levels whose two-level effect is not estimable is skipped with its reason. `ancova`
with no `covariate_cols` (or only blank names) stops before the file is read, and so does, once
the file is read, an `ancova` whose covariates are all constant on the analysed rows (a
constant covariate is dropped from the design): "ANCOVA needs at least one covariate; with
none, use anova1."

## Missing values

Leave the cell empty. A correlation or a group design tests a channel only if every analysed
subject has it. The t-tests use the subjects each channel has, and drop a channel only when it
has too few for the test to be defined under every relabelling. Channels that were not tested
are listed in the report, the workbook and the `.json` file.

## From Python

```python
from hdeeg_snpm.output import run_and_write
paths = run_and_write({"comparison": "unpairedT", "channels": "egi",
                       "data1_file": "groupA.csv", "data2_file": "groupB.csv",
                       "permutations": 5000, "output_path": "out"})
```

`hdeeg_snpm.pipeline.run(params)` returns the results without writing files.
