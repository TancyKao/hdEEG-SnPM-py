# hdeeg-snpm (Python)

Permutation statistics for high-density EEG scalp topographies: a statistic per channel,
threshold-free cluster enhancement (TFCE) and cluster correction over the channel-neighbour
graph, and a max-statistic permutation null for family-wise error control.

This is the Python port of the MATLAB toolbox [hdEEG-SnPM](https://github.com/TancyKao/hdEEG-SnPM).
The MATLAB toolbox is the reference: an analysis here counts as validated only once it reproduces
the MATLAB statistic maps and corrected p-values on shared inputs using the same permutation
indices.

**Status: v0.1.0 (phase 1) private pre-release; phase 2 (circular phase analyses) in progress and not released. It reproduces the MATLAB toolbox on the parity fixtures; use the MATLAB toolbox for published inference until the public release.**

| Piece | State |
|---|---|
| Recording systems (EGI 256 HydroCel, Compumedics 257 Neuvo) | ported; Compumedics checked against MATLAB on anova1, correlationP, pairedT, unpairedT |
| TFCE and cluster finding | ported; equal to MATLAB on 206 stored maps |
| Paired, unpaired, one-sample t-tests; Pearson and Spearman correlation with covariates | ported; reproduces MATLAB on 31 reference runs |
| Group designs: anova1, ancova, regression, rmanova, mixed2way | ported; reproduces MATLAB on 8 reference runs (37 permutation loops) |
| Results file, Excel workbook, scalp maps, HTML report, `snpm run` | written; Excel workbook matches MATLAB's numbers on 9 reference runs |
| TurtleWave `neural_events.db` reader | written; density checked only in TurtleWave's environment (below) |
| Circular phase analyses: circ_phase_group, circ_phase_group_u2, circ_corrAngLinear | ported in phase 2, not released; reproduces MATLAB on 8 reference runs |
| Mixed linear model, GUI, spectral-power folders | not ported |

How agreement with MATLAB is checked, and where the two differ on purpose:
[docs/explanation/parity-with-matlab.md](docs/explanation/parity-with-matlab.md).
All documentation: [docs/README.md](docs/README.md).

## Use

    snpm run examples/paired_t.toml

writes a results file, a summary, an Excel workbook and a self-contained HTML report.

## Install (development)

    uv venv && uv pip install -e ".[dev]"
    .venv/bin/pytest -q

Event density from TurtleWave databases is computed by TurtleWave itself, so its tests are
skipped in this environment. Before a release, run them in TurtleWave's environment from the
repository root, with TurtleWave-hdEEG checked out beside this repository:

    PYTHONPATH=src ../TurtleWave-hdEEG/.venv/bin/python -m pytest -q tests/test_turtlewave.py

Every test must pass with none skipped.
