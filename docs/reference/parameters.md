# Parameters

The same keys are used in a TOML file for `snpm run` and in the dictionary passed to
`hdeeg_snpm.pipeline.run` or `hdeeg_snpm.output.run_and_write`. Names follow the MATLAB toolbox.

| Key | Values | Default | Meaning |
|---|---|---|---|
| `comparison` | `pairedT`, `unpairedT`, `onesampleT`, `correlationP`, `correlationS`, `anova1`, `ancova`, `regression`, `rmanova`, `mixed2way`, `mixedmodel`, `circ_phase_group`, `circ_phase_group_u2`, `circ_corrAngLinear` | required | the analysis |
| `tail` | `both`, `left`, `right` | `both` | `left` and `right` exist for `pairedT` and `unpairedT` only; `mixedmodel` requires `both` and refuses anything else (`oneSidedTail`). For `left` the reported statistic has the sign of Data 2 minus Data 1, as in MATLAB |
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
| `lmm_meta_cols` | list of column names | required for `mixedmodel` | columns of the long-format file that are not channels |
| `lmm_dv`, `lmm_fixed`, `lmm_random`, `lmm_effect`, `lmm_effect_type` | see the how-to | required for `mixedmodel` | response, fixed part, random part (`(1\|<column>)` only), effect tested, `continuous` or `factor` |
| `lmm_perm` | `within_subject`, `group_label` | from `lmm_effect_type` | permutation scheme of `mixedmodel` |
| `lmm_subject`, `lmm_group` | column names | `Subject`, `group` | subject column, and the group column relabelled by `group_label` |
| `circ_units` | `rad`, `deg` | required for circular analyses, no default | units of the angle files |
| `circ_convention` | `literature_uppeak0`, `yasa_uppeak0`, `luna_zerocross0`, `turtlewave_pre_v4`, `custom` | required for circular analyses, no default | where the exported angle puts zero |
| `circ_zero_offset_deg` | number | | rotation in degrees for `custom` |
| `count1_file`, `count2_file`, `count1_sheet`, `count2_sheet` | path, sheet name | | event counts per group, required for `circ_phase_group` and `circ_phase_group_u2` |
| `measure_col` | column name | the only column | the measure of `circ_corrAngLinear` |
| `circ_count_covariate_mode` | `per_channel`, `per_subject` | `per_channel` | how log(event count) enters the circular group analyses |
| `circ_collinearity_max` | number | 0.8 | largest \|point-biserial r(log count, group)\| at which a channel is still tested |
| `circ_count_col` | column name | a covariate whose name contains `count` | the event-count covariate the Tier-1 guard looks for |

Fixed settings: significance level 0.05; TFCE with E = 0.5, H = 2 and step 0.1 (0.005 for
`circ_phase_group_u2`). The circular analyses accept only `tail = "both"` and
`datatype = "absolute"`; `mixedmodel` accepts only `tail = "both"`. How the circular keys are used:
[Run a circular (phase) analysis](../how-to/run-a-circular-analysis.md); the mixed-model keys:
[Run a mixed linear model](../how-to/run-a-mixed-model.md).

Command line overrides: `snpm run file.toml --permutations 10000 --seed 1`.
