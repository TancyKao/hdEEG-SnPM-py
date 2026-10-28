# Changelog

## 0.7.0 (2026-10-27)

Added:
- The window reads a table file as well as TurtleWave databases: "Data source" switches between
  "TurtleWave folder" and "Table file (CSV or Excel)". A table is read with the pipeline's own
  reader, an Excel workbook offers its sheets, and the window shows what it detected ("Detected:
  N channels, M other columns: …") and how many of the channels the chosen recording system
  knows. The other columns become `meta_cols` and fill the role menus. In table mode the window
  offers the paired, unpaired and one-sample t-tests, the one-way ANOVA, ANCOVA, regression and
  the Pearson and Spearman correlations; a two-file analysis takes a second file (for a
  correlation, a score table). The saved config is an ordinary one, naming `data1_file` and
  `data2_file`, or `data_file` and `meta_cols`, with no `[turtlewave]` section.
- ANCOVA (group column plus covariate columns) and regression (predictor column plus optional
  covariate columns) in the window, from either source; with TurtleWave databases the columns
  come from the subjects file. Text covariates such as sex are dummy-coded by the engine.
- Example `examples/regression.toml` with a synthetic table, `examples/data/glm_regression.csv`
  (40 made-up subjects with group, logAHI, age and sex; an effect planted on E129 and its
  neighbours), made by `examples/make_regression_example.py`. It serves the regression and the
  ANCOVA, from a config file or in the window's Table file mode.

Fixed (ported from the MATLAB reference, texts shared word for word):
- GLM presets: an effect aliased with its nuisance columns (a covariate repeating the
  predictor, a covariate splitting subjects as the groups do, an `rmanova` condition seen only
  in single-row subjects) gave a finite but arbitrary statistic through the pseudo-inverse. It
  is now refused (`glm.GLMInputError`, code `effectAliased`), naming the aliased columns; this
  applies to the main design, the three `mixed2way` effects and the simple effects. When only
  the covariates are dependent (age and age in months) the run warns
  (`glm.GLMNuisanceRankDeficient`) and gives the maps of the run without the redundant
  covariate. A post-hoc pair is skipped only when its own effect is not estimable (it used to
  be skipped on any rank deficiency) and its error degrees of freedom are n - rank(X).
  `Design.col_names` names every design column.
- `ancova` with no `covariate_cols` (absent, empty, `""` or `[""]`) is refused before the file
  is read, and an `ancova` whose listed covariates are all constant or one-level on the
  analysed rows (so the design drops every one) is refused after reading (code
  `ancovaNoCovariate`, `glm.require_covariate`; also checked by `glm.plan_preset` when called
  directly).
- Mixed model: a fixed part singular on the metadata, which left every channel without a
  statistic, is refused up front (`fixedRankDeficient`), naming the aliased coefficients.
- Mixed model: a continuous effect resolves to exactly one coefficient (exact name, the same
  `:`-parts in another order, or a one-coefficient term) instead of the first coefficient
  containing it as a substring, which silently mapped another hypothesis (a three-level
  `group` as B versus A only, `POWER` as a slope difference, `time` as `sleeptime`). A term
  with several coefficients is refused (`ambiguousEffect`). A two-level `group` now appears in
  the outputs as `group_B`. `LMM` matches its effect exactly.
- Regression with a text predictor column (such as `group`) stopped with "could not convert
  string to float". It is now refused before the design is built (`glm.GLMInputError`, code
  `predictorNotNumeric`, MATLAB `core_snpm_glm:predictorNotNumeric`), naming the column and one
  of its values and pointing to `anova1` or `ancova`. A text column is refused even when its
  values look like numbers, and so is a true/false column (MATLAB's readtable reads it as
  text; a two-level predictor is a group comparison); numeric columns pass. The window's
  predictor menu lists only numeric columns of the table or subjects file, true/false columns
  left out; the covariate list is unchanged.

## 0.6.1 (2026-10-26)

Fixed:
- Density from databases written by TurtleWave 4.3 stopped with "no denominator": TurtleWave
  4.6 looked for its current default reject set, which those databases do not store. The
  reader now passes the reject set the detection stored (for 4.3: artefacts rejected, arousals
  kept) to TurtleWave explicitly. A database storing several sets needs `reject_types` in the
  `[turtlewave]` section (`"none"` names the set that rejects nothing); a table mixing sets,
  and a stored set not in TurtleWave's key spelling, are refused. `recordings.csv` gets a
  `reject_types` column (`none` for the empty set), and the report's Source card says
  "Rejected before density". The MATLAB repository's `db_to_group_table.py` follows the same
  rules.
- `subject_from = "auto"` refused databases that store the subject with the session appended
  (`sub-01_BL` in `sub-01/ses-BL/`); that ID now agrees with its folder and the folder subject
  is used, so the sessions pair up.

Changed:
- The distribution is displayed as `hdEEG-SnPM` (pyproject `name`); PyPI normalises it to
  `hdeeg-snpm`, so it is the same project and `pip install hdeeg-snpm` keeps working. The
  import package `hdeeg_snpm`, the `snpm`/`snpm-gui` commands and the extras are unchanged.
- The window shows the reject set of density ("Rejected before density"), or a choice of it
  when the databases store several, and words a
  subject-ID error for its menu ("set Subject ID from to folder or database").

## 0.6.0 (2026-10-26)

Added:
- A window for running an analysis straight from TurtleWave databases: `snpm gui` or
  `snpm-gui`, with the optional extra `pip install "hdeeg-snpm[gui]"` (PyQt5). It lists what
  the databases hold, takes the paired or unpaired t-test, the one-way ANOVA or a Pearson or
  Spearman correlation, fills in the same settings as a `[turtlewave]` section and runs them in
  the background; refusals come from the same checks as `snpm run` and are shown with their
  text. **Save config…** writes the settings as a TOML file that `snpm run` repeats with
  identical results. Without PyQt5 the command says how to install it. Guide:
  `docs/how-to/use-the-gui.md`.

Changed:
- `[turtlewave]` with `unpairedT`: group labels in `subjects_file` are compared as text, as
  written (`01` stays `01`, `1.5` stays `1.5`), and a subject whose group cell is empty is left
  out and listed (it used to be compared as the text `nan`). Other columns are read as before.

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
