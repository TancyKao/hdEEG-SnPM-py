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

- Corrected p-values count a null value as reaching the observed value when the two agree to
  1e-12 relative (`legacy._at_least`). Known consequence: one channel in one fixture differs
  from MATLAB by one count (`KNOWN_TIES` in `tests/test_parity_legacy.py`).
- Clusters are ordered by smallest p, not by MATLAB's `dmperm` order.
- The pipeline default is 5000 permutations for every analysis.
- The HTML report is its own design and is not compared with MATLAB's.

## Project notes

Working notes live in `_scratch/` (git-ignored): `STATE.md` is the current state, `journal.md`
the dated log.

@_scratch/STATE.md

## Not ported

Circular phase tier (phase 2), mixed linear model (phase 3), GUI, spectral-power folders, the
2447-voxel source path.
