# hdEEG-SnPM (Python)

Permutation statistics for high-density EEG scalp topographies: a statistic per channel,
threshold-free cluster enhancement (TFCE) and cluster correction over the channel-neighbour
graph, and a max-statistic permutation null for family-wise error control.

This is the Python port of the MATLAB toolbox [hdEEG-SnPM](https://github.com/TancyKao/hdEEG-SnPM).
The MATLAB toolbox is the reference: an analysis here counts as validated only once it reproduces
the MATLAB statistic maps and corrected p-values on shared inputs using the same permutation
indices.

**Status: v0.8.0, beta. It reproduces the MATLAB toolbox on the parity fixtures (phases 1 to 3); the MATLAB toolbox stays the reference. Changes: [CHANGELOG.md](https://github.com/TancyKao/hdEEG-SnPM-py/blob/main/CHANGELOG.md).**

| Piece | State |
|---|---|
| Recording systems (EGI 256 HydroCel, Compumedics 257 Neuvo) | ported; Compumedics checked against MATLAB on anova1, correlationP, pairedT, unpairedT |
| TFCE and cluster finding | ported; equal to MATLAB on 206 stored maps |
| Paired, unpaired, one-sample t-tests; Pearson and Spearman correlation with covariates | ported; reproduces MATLAB on 33 reference runs |
| Group designs: anova1, ancova, regression, rmanova, mixed2way | ported; reproduces MATLAB on 8 reference runs (37 permutation loops) |
| Results file, Excel workbook, scalp maps, HTML report, `snpm run` | written; Excel workbook matches MATLAB's numbers on 9 reference runs |
| TurtleWave `neural_events.db` reader | written; density checked only in TurtleWave's environment (below) |
| Circular phase analyses: circ_phase_group, circ_phase_group_u2, circ_corrAngLinear | ported in phase 2 (v0.2.0); reproduces MATLAB on 8 reference runs |
| Mixed linear model (`mixedmodel`), random intercept `(1\|Subject)` only | ported in phase 3 (v0.3.0); agrees with MATLAB on 6 reference runs within the fit tolerances (except a few permuted fits where MATLAB's optimiser stops short), correction exact |
| Window (`snpm gui`) for TurtleWave databases or a CSV/Excel table: every analysis but the mixed model (`mixedmodel`); the circular analyses from a table only | written; MATLAB's App Designer GUI itself is not ported |
| Spectral-power folders, other random structures | not ported |

How a cluster is scored depends on the analysis:

| Analyses | Cluster statistic |
|---|---|
| Paired, unpaired, one-sample t-tests; Pearson and Spearman correlations | extent: the number of channels in the cluster |
| anova1, ancova, regression, rmanova (with post-hoc pairs), mixed2way (all effects, simple effects), circular analyses, `mixedmodel` | mass: the sum of t² or F over the cluster (`cluster_statistic = "sum"`, the default since 0.8.0), or that sum divided by the number of channels (`"mean"`, the rule before 0.8.0) |

How agreement with MATLAB is checked, and where the two differ on purpose:
[docs/explanation/parity-with-matlab.md](https://github.com/TancyKao/hdEEG-SnPM-py/blob/main/docs/explanation/parity-with-matlab.md).
All documentation: [docs/README.md](https://github.com/TancyKao/hdEEG-SnPM-py/blob/main/docs/README.md).

## Install

    pip install hdeeg-snpm

## Use

    snpm run examples/paired_t.toml

writes a results file, a summary, an Excel workbook and a self-contained HTML report. The example
files are in this repository, under `examples/`.
A `[turtlewave]` section in the configuration file runs an analysis straight from a folder of
TurtleWave `neural_events.db` files ([how](https://github.com/TancyKao/hdEEG-SnPM-py/blob/main/docs/how-to/read-turtlewave-databases.md#run-an-analysis-straight-from-the-databases)).
The same, and an analysis of a CSV or Excel table, is available in a window:
`pip install "hdeeg-snpm[gui]"`, then `snpm gui`
([how](https://github.com/TancyKao/hdEEG-SnPM-py/blob/main/docs/how-to/use-the-gui.md)).

## Install (development)

    uv venv && uv pip install -e ".[dev,gui]"
    .venv/bin/pytest -q

Event density from TurtleWave databases is computed by TurtleWave itself, so its tests are
skipped in this environment. Before a release, run them in TurtleWave's environment from the
repository root, with TurtleWave-hdEEG checked out beside this repository. First install this
package into TurtleWave's environment (this also installs openpyxl), then run both TurtleWave
test files with that environment's Python:

    ../TurtleWave-hdEEG/.venv/bin/pip install -e .
    ../TurtleWave-hdEEG/.venv/bin/python -m pytest -q tests/test_turtlewave.py tests/test_turtlewave_config.py

Every test must pass with none skipped.
