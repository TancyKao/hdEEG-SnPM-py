"""The files an analysis writes: results, Excel workbook, HTML report, and the command line."""

import json
import re

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import cli
from hdeeg_snpm.montage import get_montage
from hdeeg_snpm.output import run_and_write
from hdeeg_snpm.report.excel import SHEETS

PLANTED = ["E129", "E130", "E131", "E118", "E119", "E128", "E142", "E143"]


@pytest.fixture(scope="module")
def paired_files(tmp_path_factory):
    """24 subjects, 178 channels, an effect planted in 8 neighbouring channels; one missing cell."""
    rng = np.random.default_rng(3)
    labels = list(get_montage("egi").labels)
    d = tmp_path_factory.mktemp("in")
    a = rng.normal(10, 1, (24, len(labels)))
    b = a + rng.normal(0, 1, a.shape)
    b[:, [labels.index(c) for c in PLANTED]] += 2.0
    a[3, labels.index("E10")] = np.nan
    ids = [f"sub{i:03d}" for i in range(1, 25)]
    for name, m in (("a", a), ("b", b)):
        f = pd.DataFrame(m, columns=labels)
        f.insert(0, "Subject", ids)
        f.to_csv(d / f"{name}.csv", index=False)
    return d


@pytest.fixture(scope="module")
def written(paired_files, tmp_path_factory):
    out = tmp_path_factory.mktemp("out")
    params = {"comparison": "pairedT", "tail": "both", "channels": "egi", "datatype": "absolute",
              "data1_file": str(paired_files / "b.csv"), "data2_file": str(paired_files / "a.csv"),
              "permutations": 300, "seed": 11, "output_path": str(out)}
    return params, run_and_write(params)


def test_all_four_files_are_written(written):
    _, paths = written
    assert set(paths) == {"npz", "json", "xlsx", "html"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    assert paths["html"].name.endswith("_report.html")


def test_same_seed_gives_the_same_numbers(written):
    params, paths = written
    again = run_and_write(params)
    a, b = np.load(paths["npz"]), np.load(again["npz"])
    assert set(a.files) == set(b.files)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a.files)


def test_planted_cluster_is_recovered_and_reported(written):
    _, paths = written
    effect = json.loads(paths["json"].read_text())["effects"][0]
    assert set(PLANTED) <= set(effect["significant_channels"]["tfce"])
    top = effect["clusters"][0]
    assert top["number"] == 1 and top["p"] <= 0.05 and set(PLANTED) <= set(top["channels"])


def test_workbook_has_the_matlab_sheets_and_the_results_values(written):
    _, paths = written
    book = pd.read_excel(paths["xlsx"], sheet_name=None)
    assert tuple(book) == SHEETS
    arrays, summary = np.load(paths["npz"]), json.loads(paths["json"].read_text())
    assert list(book["statVal"].columns) == summary["channels"]
    for sheet, key in (("statVal", "main/stat"), ("pVal_or_pCritVal", "main/p_uncorrected"),
                       ("pValTFCE", "main/p_tfce")):
        np.testing.assert_allclose(book[sheet].to_numpy(float)[0], np.round(arrays[key], 4),
                                   atol=1e-12, equal_nan=True)
    sig = summary["effects"][0]["significant_channels"]
    assert list(book["correctTFCEsigch"]["ChannelLabel"]) == sig["tfce"]
    assert list(book["correctSnPMsigch"]["ChannelLabel"]) == sig["cluster"]
    n = book["effectiveN"].set_index("Metric")
    assert n.loc["n_analysed", "E10"] == 23 and n.loc["n_analysed", "E1"] == 24


def test_report_is_self_contained_and_states_what_the_results_file_states(written):
    params, paths = written
    page = paths["html"].read_text(encoding="utf-8")
    summary = json.loads(paths["json"].read_text())
    effect = summary["effects"][0]
    assert not re.search(r'(src|href)="https?:|<script|<link', page)
    assert 'src="data:image/png;base64,' in page
    assert f"Random seed {params['seed']}" in page and summary["version"] in page
    assert f"{len(effect['significant_channels']['tfce'])} after TFCE correction" in page
    assert f"over {effect['global']['n_channels']} channels" in page
    assert f"{effect['global']['statistic']:.4g}" in page
    for c in effect["clusters"]:
        assert re.search(rf'<td class="num">{c["number"]}</td><td class="num">{len(c["channels"])}'
                         rf'</td><td class="num">{c["statistic"]:.4g}</td>'
                         rf'<td class="num[^"]*">{c["p"]:.4f}</td>', page)
    assert "number of channels" in page          # the cluster statistic is disclosed


def test_command_line_runs_a_toml_file(paired_files, tmp_path, capsys):
    cfg = tmp_path / "run.toml"
    cfg.write_text(f'comparison = "unpairedT"\nchannels = "egi"\n'
                   f'data1_file = "{paired_files / "a.csv"}"\n'
                   f'data2_file = "{paired_files / "b.csv"}"\npermutations = 50\nseed = 1\n')
    assert cli.main(["run", str(cfg)]) == 0
    printed = capsys.readouterr().out
    assert "_report.html" in printed and (tmp_path / "snpm_output").is_dir()
    assert cli.main(["run", str(cfg), "--permutations", "0"]) == 1
    assert "permutations must be" in capsys.readouterr().err


def test_command_line_reports_a_bad_file_without_a_traceback(tmp_path, capsys):
    cfg = tmp_path / "bad.toml"
    cfg.write_text('comparison = "pairedT"\ndata1_file = "nope.csv"\ndata2_file = "nope.csv"\n')
    assert cli.main(["run", str(cfg)]) == 1
    assert "snpm:" in capsys.readouterr().err
