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

    uv venv --python 3.10 && uv pip install -e ".[dev,gui]"   # gui: PyQt5, for the window tests
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
- `circ.py`: circular tier statistics: angle import and zero convention, inversion check,
  Hotelling T² (observed and permuted), Watson's U², circular-linear F, the per-channel
  event-count covariate, the Tier-1 linearisation and its provenance stamp, the descriptive panel.
- `circ_pipeline.py`: circular files to results (`run_circ`): reading and aligning angle, count
  and measure files, the estimability gate fixed before the loop, the permutation driver, and
  `tier1_guard`, which `pipeline.py` runs before every non-circular analysis.
- `lmm.py`, `lmm_pipeline.py`: mixed linear model tier, random intercept `(1|Subject)` only
  (closed-form maximum-likelihood fit, `within_subject` and `group_label` schemes, rows with a
  missing model variable (NaN, empty, +-Inf) dropped up front; settings that leave nothing to
  permute, and an effect that is not a coefficient or term of the fixed part, refused before any
  fit; channels constant within each subject (within-subject share <= 1e-6) excluded once when
  they are the response or the `within_subject` predictor, and flat channels under `group_label`
  with a metadata response; a per-labelling backstop gives no
  statistic to a fit with no residual variation or a moment subject/residual SD ratio >= 1e3).
- `missing.py`: which channels can be tested. `pipeline.py`: files to results.
- `views.py`, `output.py`, `report/`: one description of an effect, then the writers.
- `io/tables.py`, `io/turtlewave.py`: inputs. `io/turtlewave_config.py`: the `[turtlewave]`
  section of a run config (databases to the analysis files). `cli.py`: `snpm run config.toml`.
- `gui/form.py` (no Qt): the window's choices as `[turtlewave]` run-config params and TOML.
  `gui/window.py` (the only PyQt5 import): a thin view that calls `output.run_and_write` in a
  `QThread`; it checks nothing itself. `snpm gui` / `snpm-gui`; extra `[gui]`. Tests run with
  `QT_QPA_PLATFORM=offscreen`; `tools/gui_screenshot.py` remakes `docs/images/gui-window.png`.

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
  a `posthoc` sheet); the mixed model adds a `mixedModel` sheet (formula, random structure,
  effect, scheme, trials and subjects analysed, cluster statistic, whole-head test) that MATLAB
  does not write; its `droppedRows` sheet follows MATLAB. Numbers, channel sets and
  reason texts are identical (`tests/test_parity_workbook.py`).
- Mixed linear model fit: Python's estimator (closed form in the subject-to-residual SD ratio,
  `lmm.fit_random_intercept`) always reaches the true maximum-likelihood optimum, including a
  subject SD of exactly 0. MATLAB's `fitlme` can stall at a subject SD of 0 when the optimum is
  above it; MATLAB now refits those channels (guard in `snpm_lmm_fitlme.m`). Permuted refits
  agree within the fit tolerance except where `fitlme` stops short of the maximum, where this
  package's likelihood is higher: 9 of 400 maps in `lmm_degenerate`, worst 1.6e-4 of the map
  maximum (`KNOWN_MATLAB_OPTIMISER_SHORTFALLS` in `tests/test_parity_lmm.py`); the effect on
  inference is negligible. Tolerances: `docs/explanation/parity-with-matlab.md`.
- Mixed linear model formula: the fixed part refuses removing the intercept (`- 1`, `0`), which
  MATLAB `fitlme` accepts.
- TurtleWave reader (`io/turtlewave.py`) versus the MATLAB repository's `db_to_group_table.py`:
  a database older than TurtleWave 4.3 has no density in the reader, while the bridge divides by
  hypnogram time when given `--allow-hypnogram-denominator`; the reader names a subject by the
  ID stored in the database when there is one (by the folder when that ID is the folder subject
  plus the session), the bridge by the folder name; the reader writes
  the channels of the band asked for, the bridge every channel seen in any band; a recording in
  which the band was never detected is an error in the reader and an empty row (with a
  warning) in the bridge. Explained in `docs/how-to/read-turtlewave-databases.md`.
- Circular tier, reasons for untested channels: the workbook, the HTML report and
  `excluded_reasons` in the `.json` give each channel the route that excluded it (fewer than 8
  subjects per group; log(count) collinear with group; incomplete column under the per-channel
  covariate; for `circ_corrAngLinear`, fewer than 8 subjects with both an angle and a measure
  value), in the wording of the MATLAB parity export. MATLAB's own outputs carry one string for
  every excluded channel (`core_snpm_circ.m`; for the two group analyses it combines the three
  routes); the `.json` keeps that string word for word as `circular.exclusion_reason`. The
  channel sets are identical.

## Project notes

Working notes live in `_scratch/` (git-ignored): `STATE.md` is the current state, `journal.md`
the dated log.

@_scratch/STATE.md

## Not ported

MATLAB's App Designer GUI (the window in `gui/` covers TurtleWave databases only), spectral-power
folders, the 2447-voxel source path. In the mixed linear model, any random
structure other than one random intercept, `(1|G)`.
