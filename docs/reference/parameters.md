# Parameters

The same keys are used in a TOML file for `snpm run` and in the dictionary passed to
`hdeeg_snpm.pipeline.run` or `hdeeg_snpm.output.run_and_write`. Names follow the MATLAB toolbox.

| Key | Values | Default | Meaning |
|---|---|---|---|
| `comparison` | `pairedT`, `unpairedT`, `onesampleT`, `correlationP`, `correlationS`, `anova1`, `ancova`, `regression`, `rmanova`, `mixed2way` | required | the analysis |
| `tail` | `both`, `left`, `right` | `both` | `left` and `right` exist for `pairedT` and `unpairedT` only. For `left` the reported statistic has the sign of Data 2 minus Data 1, as in MATLAB |
| `channels` | `egi`, `compu` | `egi` | recording system: EGI 256 HydroCel (178 scalp channels analysed) or Compumedics 257 Neuvo (249) |
| `datatype` | `absolute`, `logscale`, `normalize` | `absolute` | no transform, log10, or z-score across channels within each subject |
| `permutations` | whole number, at least 20 | 5000 | permutations per test. When the number of possible relabellings is no larger than this, all of them are enumerated instead of drawn at random |
| `seed` | whole number | drawn and recorded | seed of the random generator |
| `output_path` | folder | `snpm_output` beside the TOML file | where files are written |
| `data1_file`, `data2_file` | path | | the two files of a t-test or correlation |
| `data1_sheet`, `data2_sheet`, `data_sheet` | sheet name | first sheet | for Excel files |
| `use_covariates`, `covariate_file` | `true`, path | off | covariates for a correlation |
| `data_file` | path | | the single file of a group design |
| `meta_cols` | list of column names | required for group designs | columns that are not channels |
| `group_col`, `condition_col`, `subject_col`, `predictor_col`, `covariate_cols` | column names | | roles of the design columns |

Fixed settings: significance level 0.05; TFCE with E = 0.5, H = 2 and step 0.1.

Command line overrides: `snpm run file.toml --permutations 10000 --seed 1`.
