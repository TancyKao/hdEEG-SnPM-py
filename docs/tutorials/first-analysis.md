# Run your first analysis

This walks through a paired t-test on the example data that ships with the MATLAB toolbox
(24 subjects, two conditions, EGI 256 net). It takes about a minute.

## 1. Install

    uv venv && uv pip install -e .

## 2. Describe the analysis in a file

`examples/paired_t.toml`:

```toml
comparison   = "pairedT"
tail         = "both"
channels     = "egi"
datatype     = "absolute"
data1_file   = "../../SnPM_2025/example_data/paired_condA.csv"
data2_file   = "../../SnPM_2025/example_data/paired_condB.csv"
permutations = 1000
seed         = 20261013
output_path  = "../_scratch/example_output"
```

Each data file has one row per subject, a `Subject` column, and one column per channel named
as the recording system names it (`E1`, `E2`, ...). Paths are relative to the TOML file.

## 3. Run it

    snpm run examples/paired_t.toml

Four paths are printed: a results file (`.npz`), a summary (`.json`), an Excel workbook
(`.xlsx`) and a report (`_report.html`).

## 4. Read the report

Open the `_report.html` file in a browser. It is one file with the images inside it, so it can
be emailed or archived as it is. It shows the t map, the channels that survive TFCE correction,
and the significant clusters, numbered to match the cluster table. In this example the planted
effect appears as one cluster of 19 channels right of the midline behind the vertex.

Run the same file again and the numbers are identical, because the seed is fixed. Leave `seed`
out and a seed is drawn and written into the report and the `.json` file, so the run can still
be repeated.
