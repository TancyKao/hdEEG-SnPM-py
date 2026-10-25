# Changelog

## 0.5.0 (2026-10-25)

Added:
- A `[turtlewave]` section in the `snpm run` configuration file: one command reads a folder of
  TurtleWave `neural_events.db` files, builds the table of one measure, splits or joins it for
  the analysis (`sessions`, `groups`, `subjects_file`, `behavior_col`) and runs it. The tables
  and `recordings.csv` are kept in a `_tables` folder named after the run; the summary file and
  the report record the source. Subjects on only one side of the join are refused (IDs compared
  as text, ignoring a leading `sub-` and case); subjects left out of a paired or two-group
  comparison are listed; recordings from several sessions are not pooled into a one-recording
  analysis unless `sessions` names one. `mixedmodel` and the circular analyses are refused.
  Example: `examples/turtlewave_unpaired.toml`.

Changed:
- CSV files are read without the `float_precision` argument under pandas 3, whose default
  parser is already exact; numbers are unchanged.

## 0.4.0 (2026-10-24)

First public release, and the first on PyPI.

Changed, in step with the MATLAB toolbox:
- The corrected p of the t-tests and correlations is now b/N. The observed labelling is one of
  the N permutations and is counted once; it used to be counted twice, which made p slightly
  conservative (by less than 1/(N+1)); for a one-sided paired test on 5 subjects the smallest p
  was 2/33 instead of 1/32, so it could not reach 0.05.
  Group designs, mixed models and circular analyses keep (b+1)/(N+1). A supplied `perms` matrix
  without the observed labelling is refused.
- Mixed model: +-Inf in a model column counts as missing; an effect naming no coefficient or term
  is refused, listing the available ones; a within-subject shuffle of an effect that is constant
  within subject is refused; under group relabelling a channel with one value on every trial is
  excluded and a response constant within subject is refused; reference levels and coefficient
  names follow MATLAB's fitlme (text columns not converted to categorical keep their order of
  first appearance).
- circ_corrAngLinear: the exclusion reason no longer mentions groups.

## 0.3.0 (2026-10-20)

Mixed linear model (`mixedmodel`) with a random intercept per subject.

## 0.2.0 (2026-10-19)

Circular phase analyses: circ_phase_group, circ_phase_group_u2, circ_corrAngLinear.

## 0.1.0 (2026-10-18)

t-tests, Pearson and Spearman correlation with covariates, group designs (anova1, ancova,
regression, rmanova, mixed2way), results file, Excel workbook, scalp maps and HTML report, and
an optional reader for TurtleWave event databases.
