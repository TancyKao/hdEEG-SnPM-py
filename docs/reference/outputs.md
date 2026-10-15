# Output files

Every run writes four files named `<data file>_<comparison>_<date>_<time>`.

## `.json`: what was run and what was found

Parameters, seed, package version, recording system and channel list, notes and warnings, and
for each effect: the whole-head test, the channels not tested and why, the significant channels
(uncorrected, TFCE-corrected, in significant clusters), and every cluster with its number,
members, statistic and corrected p-value. Post-hoc contrasts sit under the effect they follow.

## `.npz`: the maps

Arrays named `<effect>/<name>`, one value per analysed channel in the order of `channels` in the
`.json` file: `stat`, `p_uncorrected`, `tfce`, `p_tfce`, `excluded` (channel indices), and
`cluster<k>` (channel indices of cluster k). The t-test and correlation analyses add `p_max`
(max-statistic corrected), `n_analysed` and `n_available`. Indices count from 0.

Effect names: `main` for single-effect analyses; `group`, `condition`, `interaction`,
`condition@<group>` for `mixed2way`; post-hoc contrasts are `<effect>__<A>_vs_<B>`.

## `.xlsx`: the MATLAB workbook layout

Sheets `statVal`, `pVal_or_pCritVal`, `pValTFCE`, `effectiveN`, `excludedChannels`,
`uncorrectsigch`, `correctTFCEsigch`, `correctSnPMsigch`, values rounded to 4 decimals. Channel
numbers in this file count from 1, as in MATLAB, and the label is given beside each. Analyses
with several effects add the same sheets with a number prefix and a `contents` sheet.
`effectiveN` is written for the t-test and correlation analyses.

## `_report.html`

One static page with the images embedded; nothing is loaded from the network. Per effect: the
statistic map, TFCE-corrected channels, numbered significant clusters with a table, the
channels not tested, and the whole-head test. The whole-head test is a parametric test on the
channel average and is not corrected; it is descriptive.
