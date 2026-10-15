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
| Corrected p-values | identical, except as listed under "Ties" |
| Clusters | same channels, same statistic, same p-value |
| Whole-head test | relative difference below 1e-10 (statistic), 1e-8 (p) |

## What has been checked (14 October 2026)

39 reference runs, all passing:

- 31 t-test and correlation runs: paired, unpaired and one-sample t-tests (both tails, and left
  and right for paired and unpaired); Pearson and Spearman correlation, channel against channel
  and against one behavioural score, with and without covariates; small samples where every
  relabelling is enumerated; data with missing cells, a channel missing in one group, and a
  subject with a missing covariate; ties in Spearman data; `logscale` and `normalize`; both
  recording systems.
- 8 group-design runs covering 37 permutation loops: `anova1` (both recording systems),
  `ancova` with two and three groups, `regression`, `rmanova`, and `mixed2way` 2 x 2 and 2 x 3
  with every post-hoc contrast and simple effect.

The TFCE function alone is also compared with 206 stored MATLAB maps and agrees to about 1e-15.

Run it with `pytest -m parity`. The reference files are built on the machine that has MATLAB
and are not in the repository, so these tests are skipped elsewhere.

## Ties

TFCE values and cluster masses are sums of a few terms on a coarse grid, so an observed value is
often mathematically equal to a value in the null distribution. Whether such a null value counts
towards the p-value then depends on the last bit of a floating-point sum. This package counts a
null value as reaching the observed value when the two agree to one part in 10^12. That can only
raise a p-value, never lower it.

With that rule the corrected p-values equal MATLAB's at every channel of every reference run
except one: channel 78 of `pairedT_logscale`, where the observed TFCE value is 5.525000000000001
and a null value is 5.525. MATLAB does not count that permutation and this package does, so the
p-values are 0.9840 and 0.9860.

## Differences on purpose

- **Ties**, as above.
- **Default number of permutations** is 5000 for every analysis.
- **Channels with no variance** (identical values in both conditions, for example) are excluded
  and listed. A non-finite cell, such as the log of zero power, is treated as a missing cell in
  the t-tests.
- **A missing value in a design column** stops the run and names the column.
- **Cluster order**: clusters are numbered by p-value, smallest first.
- **The report** is this package's own design.
- **MATLAB fixed alongside**: `ClusterEnhancement.m` returned a result of the wrong length for a
  map with every value below the first threshold. It was corrected in MATLAB on 13 October 2026
  and both now return zeros.

## Behaviour copied from MATLAB that you should know about

- For `tail = "left"` the statistic reported is Data 2 minus Data 1. The report says so.
- The cluster-forming threshold for a correlation is chosen from the number of matched subjects
  before any subject with a missing covariate is dropped.
- The cluster-forming threshold of a one-tailed t-test is the two-sided critical t.
- With covariates, the uncorrected per-channel p-value of a correlation uses n - 2 degrees of
  freedom. Corrected p-values do not depend on it.
- The cluster statistic of the group designs is the sum of F (or t squared) over the cluster
  divided by the number of channels in the cluster.
- `normalize` z-scores across the selected scalp channels in the t-test and correlation
  analyses, and across every data column in the group designs.

## Defects shared with MATLAB, found in review and fixed (14 October 2026)

Agreement with MATLAB cannot detect a fault that both share. An independent review found two,
a follow-up found a third, and all three are fixed in both this package and MATLAB. The
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

False-positive rates after the fixes, same simulation set-up as below:

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
size is 0.036 to 0.064. Results are in [fwe_2026-10-13.csv](fwe_2026-10-13.csv).

| Analysis | TFCE | Cluster |
|---|---|---|
| pairedT | 0.045 | 0.022 |
| pairedT, 20% of cells missing | 0.041 | 0.036 |
| onesampleT | 0.053 | 0.042 |
| unpairedT | 0.042 | 0.035 |
| unpairedT, one group missing cells in half the channels | 0.050 | 0.035 |
| correlationP | 0.036 | 0.030 |
| correlationP, scattered missing cells | 0.059 | 0.041 |
| correlationS | 0.066 | 0.042 |
| anova1 (3 groups) | 0.046 | 0.057 |
| ancova (covariate with a real effect) | 0.051 | 0.040 |
| regression (nuisance correlated with the predictor) | 0.045 | 0.058 |
| rmanova (3 conditions) | 0.059 | 0.059 |
| mixed2way group | 0.036 | 0.036 |
| mixed2way condition | 0.070 | 0.061 |
| mixed2way interaction | 0.046 | 0.053 |

Two TFCE rates fall just above the range. Among 30 rates one or two are expected to by chance.
Both were repeated with 4000 new simulations: correlationS 0.049 (cluster 0.036) and mixed2way
condition 0.050 (cluster 0.054). The cluster test of the t-tests and correlations is
conservative because cluster size is a small whole number and ties count against significance.

The `mixed2way` rows in this table were run before the fixes above; the table in the previous
section has the rates after them. Not yet simulated: correlation with covariates, one-sided
tails, and samples small enough for every relabelling to be enumerated.
