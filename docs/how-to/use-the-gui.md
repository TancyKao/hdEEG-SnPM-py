# Use the window (GUI)

The window runs an analysis straight from a folder of TurtleWave `neural_events.db` files,
without writing a configuration file by hand. It fills in the same settings as a
`[turtlewave]` section of a `snpm run` configuration file and runs them the same way, so a
result from the window equals the result of `snpm run` on the configuration it saves. Every
check of the request is the one described in
[Read TurtleWave databases](read-turtlewave-databases.md#run-an-analysis-straight-from-the-databases),
and a refusal is shown with the same text.

This version reads TurtleWave databases only, and offers the paired t-test, the unpaired
t-test, the one-way analysis of variance (ANOVA) and the Pearson and Spearman correlations.
Other analyses and tables of your own still go through a configuration file.

## Install and open

The window uses PyQt5, the same toolkit as TurtleWave, so in TurtleWave's environment it is
usually installed already. Otherwise:

    pip install "hdeeg-snpm[gui]"

Open it with either command:

    snpm gui
    snpm-gui

Without PyQt5 both stop with the install command above.

![The window on a synthetic study, set up for an unpaired t-test](../images/gui-window.png)

## Fill it in, top to bottom

1. **Study.** Choose the folder that holds the databases with Browse…, or type it and press
   Enter or Load; every `neural_events.db` below it is read. A path typed but not yet read is
   not used, and Run stays greyed out until it is. The line under it says how many databases, subjects and sessions were found, and the
   table lists each database's subject, session and where the subject ID came from. Check it
   by eye. If the subject IDs are wrong, change "Subject ID from" (auto, database or folder
   name; see [which subject a database belongs to](read-turtlewave-databases.md#which-subject-a-database-belongs-to)).
2. **Events.** Event type, method, stages and band list what the databases hold for that event
   type. All stages found are ticked; untick the ones to leave out (at least one must stay
   ticked). A detection run stored without a band shows no band. Measure is amplitude,
   duration, count or density. Density is computed by TurtleWave itself, so it is greyed out
   when TurtleWave is not installed.
3. **Analysis.** The analysis, the tail (t-tests only), the number of permutations, the seed,
   the recording system and the data type. When the seed is left empty, the window draws one
   when you press Run or Save config… and shows it in the field, so the saved config repeats
   the run exactly. These are the keys of the same names in [Parameters](../reference/parameters.md).
4. **Roles.** What the analysis needs besides the databases:
   - paired t-test: the two sessions, the first is Data 1;
   - unpaired t-test: a subjects file (a `Subject` column plus one column per variable; type
     the path and press Enter or Load, or use Browse…), its group column and the two groups
     to compare, the first is Data 1. Group labels are compared as written in the file, so
     `01` stays `01`; a subject with an empty group cell is left out and listed in the log;
   - one-way ANOVA: a subjects file and its group column;
   - correlations: a subjects file and the score column to correlate with;
   - any analysis other than the paired t-test, when the databases come from more than one
     session: the session to analyse.
5. **Output.** The folder for the results.

**Run** stays greyed out until everything above is filled in; the line under it says what is
still missing. The analysis runs in the background, so the window stays responsive, but its
inputs are locked until the run ends. Closing the window during a run does not stop it: the
window stays open until the run has finished and then closes, so no output is left half
written. Warnings and notes (for example subjects left out of the comparison) appear in the log. If the analysis is
refused, a message gives the reason; correct the input and press Run again.

## Where the results go

The output folder gets the four files of any run (results, summary, Excel workbook and HTML
report, described in [Output files](../reference/outputs.md)) and a folder ending in
`_tables` with the tables built from the databases and `recordings.csv`. When the run ends,
the window lists the four files, and **Open report** opens the HTML report in the browser.

## Repeat the run from the command line

**Save config…** writes the settings as a TOML file, with the full paths of this computer.
On another computer, such as a computing cluster, edit `folder`, `subjects_file` and
`output_path` in the file to where the data are there. Run it with:

    snpm run saved.toml

With the same seed the results are identical to the window's.
