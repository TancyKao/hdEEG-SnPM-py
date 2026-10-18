# CLAUDE.md

Python port of the MATLAB toolbox hdEEG-SnPM (`../SnPM_2025`, public at
github.com/TancyKao/hdEEG-SnPM): permutation statistics for high-density EEG scalp maps, with
threshold-free cluster enhancement (TFCE) and cluster correction over the channel-neighbour graph.

## Rule that governs everything here

**MATLAB is the reference.** An analysis is trusted only when, given MATLAB's own permutation
indices, it reproduces MATLAB's observed map, TFCE map, sorted null distributions, corrected
p-values and clusters on the shared fixtures. Parity goes through exported index matrices, never
through shared seeds: MATLAB's `randperm` cannot be replayed from numpy. Every engine therefore
takes a `perms=` argument. A change that breaks `pytest -m parity` is wrong until shown otherwise;
if MATLAB is the one at fault, raise it and fix it there too.

Work as a sleep-EEG statistician first: the permutation scheme must match the effect, the
observed map and every permuted map must cover the same channels on the same subjects, and
nothing may be excluded silently.

## Commands

    uv venv --python 3.10 && uv pip install -e ".[dev]"
    .venv/bin/python -m pytest -q            # everything (about 90 s)
    .venv/bin/python -m pytest -m parity -q  # MATLAB parity only
    .venv/bin/ruff check .
    .venv/bin/snpm run examples/paired_t.toml
    .venv/bin/python tools/fwe_simulation.py --sims 1000 --perms 199   # false-positive rates, ~30 min

## Layout

- `src/hdeeg_snpm/montage.py`, `assets/*.npz`: recording systems (built from the MATLAB `.mat`
  assets by `tools/convert_matlab_assets.py`). Neighbours are 0-based, -1 padded.
- `tfce.py`, `clusters.py`, `perm.py`: TFCE integral, connected components, permutation indices.
- `legacy.py`: t-test and correlation engines, whole-head test, MATLAB's Spearman p-value rule.
- `glm.py`: designs, vectorised least squares, Freedman-Lane, the loop plan of each preset.
- `missing.py`: which channels can be tested. `pipeline.py`: files to results.
- `views.py`, `output.py`, `report/`: one description of an effect, then the writers.
- `io/tables.py`, `io/turtlewave.py`: inputs. `cli.py`: `snpm run config.toml`.

## Fixtures

`parity_fixtures/` is git-ignored and is copied from `../SnPM_2025/test_fixtures/parity/`, which
is built by `scripts/export_parity_fixtures.m` in MATLAB. Field dictionary:
`parity_fixtures/README.md`. The legacy parity tests read the input files by the absolute paths
stored in the fixtures and skip when those are absent, so full parity runs only on the machine
that holds `SnPM_2025`. `tests/data/ce_golden_178.mat` is tracked.

## Deliberate differences from MATLAB

This is the single list; `docs/explanation/parity-with-matlab.md` explains each item and adds
none of its own.

- Clusters are ordered by smallest p, not by MATLAB's `dmperm` order.
- The pipeline default is 5000 permutations for every analysis.
- The HTML report is its own design and is not compared with MATLAB's.
- A missing value in a group-design column (group, condition, subject, predictor or covariate
  of `anova1`, `ancova`, `regression`, `rmanova`, `mixed2way`) stops the run and names the
  column and rows (`glm._no_missing`). MATLAB has no such check: `snpm_glm_design.m` passes the
  NaN or undefined category into the design matrix. The correlation path agrees with MATLAB
  (a subject with a missing covariate is dropped with a warning).
- Excel layout (Tancy, 2026-10-17): significant-channel sheets have one row per channel with
  index and label (MATLAB: one row of indices); `excludedChannels` names the analysis in the
  sheet prefix (MATLAB: an `Analysis` column); post-hoc status is in the `contents` sheet (MATLAB:
  a `posthoc` sheet). Numbers, channel sets and reason texts are identical
  (`tests/test_parity_workbook.py`).
- TurtleWave reader (`io/turtlewave.py`) versus the MATLAB repository's `db_to_group_table.py`:
  a database older than TurtleWave 4.3 has no density in the reader, while the bridge divides by
  hypnogram time when given `--allow-hypnogram-denominator`; the reader names a subject by the
  ID stored in the database when there is one, the bridge by the folder name; the reader writes
  the channels of the band asked for, the bridge every channel seen in any band; a recording in
  which the band was never detected is an error in the reader and an empty row (with a
  warning) in the bridge. Explained in `docs/how-to/read-turtlewave-databases.md`.

## Project notes

Working notes live in `_scratch/` (git-ignored): `STATE.md` is the current state, `journal.md`
the dated log.

@_scratch/STATE.md

## Not ported

Circular phase tier (phase 2), mixed linear model (phase 3), GUI, spectral-power folders, the
2447-voxel source path.
