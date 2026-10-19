# Output files

Every run writes four files named `<data file>_<comparison>_<date>_<time>`.

## `.json`: what was run and what was found

Parameters, seed, package version, recording system and channel list, notes and warnings, and
for each effect: the whole-head test, the channels not tested and why, the significant channels
(uncorrected, TFCE-corrected, in significant clusters), and every cluster with its number,
members, statistic and corrected p-value. Post-hoc contrasts sit under the effect they follow.
The circular analyses have no whole-head test (its fields are empty) and add a `circular`
block: the angle import and rotation applied, the event-count covariate report, the inversion
check, group sizes, subjects analysed and dropped, the permutation type and the descriptive
panel. The mixed model adds a `mixed_model` block: formula, random structure, effect and
effect type, permutation scheme, coefficient names, trials and subjects analysed, residual
degrees of freedom, the rows dropped for a missing model variable, and the number of
permutations that gave no finite statistic.

## `.npz`: the maps

Arrays named `<effect>/<name>`, one value per analysed channel in the order of `channels` in the
`.json` file: `stat`, `p_uncorrected`, `tfce`, `p_tfce`, `excluded` (channel indices), and
`cluster<k>` (channel indices of cluster k). The t-test and correlation analyses add `p_max`
(max-statistic corrected); they and the circular analyses add `n_analysed` and `n_available`.
Indices count from 0.

The circular analyses add `evaluable`, `usecov`, `collinear`, `n_finite` and
`mean_direction_deg` (one row per group); the two-group ones also `n_group1`, `n_group2` and
`point_biserial`, and `circ_corrAngLinear` adds `rho`.

The mixed model adds `evaluable`, `within_ratio` (each channel's within-subject share of
variation, NaN for an incomplete channel; at most 1e-6 means constant within each subject),
`wald`, `beta` (one row per channel, one column per fixed
coefficient), `se`, `loglik`, `sd_subject`, `sd_residual`, `boundary` (subject variance fitted as
zero), `tfce_null`, `cluster_mass_null` and `dropped_rows` (row indices of the file as read).
A mixed-model channel that is not tested carries one of four reasons in the `excludedChannels`
sheet, the report and the `.json` file, in MATLAB's words: "missing data: channel not complete
across all analysed rows (trials), so the fitted row set would differ from the permuted one";
"channel value constant within each subject: ..." (the channel is the response); "predictor
constant within each subject: ..." (the channel is the `within_subject` predictor); or "perfect
fit, statistic undefined" (the observed fit had no residual variation left). The full texts are
in [Run a mixed linear model](../how-to/run-a-mixed-model.md).

Effect names: `main` for single-effect analyses; `group`, `condition`, `interaction`,
`condition@<group>` for `mixed2way`; post-hoc contrasts are `<effect>__<A>_vs_<B>`.

## `.xlsx`: the MATLAB workbook layout

Sheets `statVal`, `pVal_or_pCritVal`, `pValTFCE`, `effectiveN`, `excludedChannels`,
`uncorrectsigch`, `correctTFCEsigch`, `correctSnPMsigch`, values rounded to 4 decimals. Channel
numbers in this file count from 1, as in MATLAB, and the label is given beside each. Analyses
with several effects add the same sheets with a number prefix and a `contents` sheet.
`effectiveN` is written for the t-test, correlation and circular analyses. The mixed model adds
`droppedRows` (as MATLAB: a summary line, then the number of rows missing each model variable)
and `mixedModel`, an item and value list that MATLAB does not write: formula, random structure,
effect, statistic, residual degrees of freedom, permutation scheme, trials and subjects analysed,
cluster statistic and whole-head test.

## `_report.html`

One static page with the images embedded; nothing is loaded from the network. Per effect: the
statistic map, TFCE-corrected channels, numbered significant clusters with a table, the
channels not tested, and the whole-head test. The whole-head test is a parametric test on the
channel average and is not corrected; it is descriptive. The circular analyses have no
whole-head test; they show a descriptive panel per group instead (mean direction, circular
standard deviation, R, Rayleigh test), which carries no inference.
