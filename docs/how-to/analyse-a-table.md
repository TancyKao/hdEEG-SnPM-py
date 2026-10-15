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
  subject ID, then one column per covariate) to remove the covariates from both sides first.

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
