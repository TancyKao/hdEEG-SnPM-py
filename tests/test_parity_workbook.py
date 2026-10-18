"""Excel workbook against MATLAB's, sheet by sheet (plan step 8).

MATLAB ran ``core_snpm_analysis`` end to end for each case with the permutation recorder on and
kept the ``.xlsx`` it wrote (``parity_fixtures/workbooks/``, see the README there). Python runs
the same input files with the recorded indices, writes its workbook, and every sheet MATLAB
wrote is compared with the Python sheet of the same name.

MATLAB's ``func_genSnpmTable`` writes the maps of ONE effect: the main effect, or for mixed2way
the interaction (``core_snpm_glm`` points the top-level fields there), plus a ``posthoc``
summary. Python writes every effect and post-hoc contrast, the first plainly named and the rest
with a numeric prefix listed on a ``contents`` sheet. The sheets MATLAB has no counterpart for
are compared with the maps MATLAB kept in its results (the ``.mat`` fixture), rounded the way
the workbook rounds them.

Numbers: both writers round to 4 decimals and the cells must be equal. The statistics agree to
1e-10, not exactly, so a value within that of a rounding boundary could round apart; none does
on these fixtures, and the failure message gives MATLAB's unrounded value if one ever does.

Layout differences that are deliberate on the Python side, and how each is compared:

- per-channel sheet headers: both write channel labels (MATLAB wrote ``Var1..VarN`` on
  ``pVal_or_pCritVal`` until SnPM_2025 commit abbdb76); values are compared cell by cell in
  channel order.
- significant-channel sheets: MATLAB writes one row of 1-based indices under ``<name>1..K``;
  Python writes one row per channel with ``ChannelIndex`` and ``ChannelLabel``. The index lists
  must be equal; the cluster sheet is compared as a set, since MATLAB concatenates clusters in
  ``dmperm`` order and Python by smallest p (CLAUDE.md, deliberate differences).
- ``excludedChannels``: MATLAB adds an ``Analysis`` column naming the loop, which Python puts in
  the sheet prefix instead. Reason texts must be identical.
- ``posthoc``: MATLAB's summary sheet is compared with Python's ``contents`` sheet and with the
  number of rows in each prefixed significant-channel sheet. Python has no ``Rows`` or
  ``Subjects`` column; the pair's ``RunOn`` text carries the subject counts.
"""

from pathlib import Path

import numpy as np
import openpyxl
import pytest
import scipy.io as sio

import parity_util as pu
from hdeeg_snpm.pipeline import run_glm, run_legacy
from hdeeg_snpm.report import excel
from hdeeg_snpm.views import effect_views

pytestmark = pytest.mark.parity

WORKBOOKS = pu.FIXTURES / "workbooks"
CASES = ["pairedT_both", "unpairedT_both", "correlationP_wide_cov", "correlationS_single",
         "miss_correlationP_scattered", "miss_unpairedT_group_channel", "anova1", "rmanova",
         "mixed2way_2x2"]

MAP_SHEETS = {"statVal": ("T", "real_T"), "pVal_or_pCritVal": ("p", "real"),
              "pValTFCE": ("p", "correctedTFCE")}
SIG_SHEETS = {"uncorrectsigch": "uncorrsigch", "correctTFCEsigch": "correctTFCEsigch",
              "correctSnPMsigch": "SnPMsigch"}
NO_SIG = "No significant chans"


def _load(case):
    path = WORKBOOKS / f"{case}.mat"
    if not path.exists() or not path.with_suffix(".xlsx").exists():
        pytest.skip(f"{path} is absent (built by SnPM_2025/_scratch/workbook_parity/"
                    "export_workbook_fixtures.m in MATLAB)")
    return sio.loadmat(path, simplify_cells=True)


def _sheets(path) -> dict[str, list[list]]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    out = {ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets}
    wb.close()
    return out


def _python_workbook(fx, out_dir: Path):
    from test_parity_glm import _glm_params
    from test_parity_legacy import _params, _perms

    logs = pu.as_list(fx["perm_log"])
    if fx["tier"] == "legacy":
        res = run_legacy(_params(fx), perms_tfce=_perms(logs[0]), perms_cluster=_perms(logs[1]))
        labels = res.prepared.setup.labels
    else:
        params = _glm_params(fx)
        if not Path(params["data_file"]).exists():
            pytest.skip("input file is not on this machine")
        res = run_glm(params, perms=[pu.index0(lg["indices"]) for lg in logs])
        labels = res.setup.labels
    views = effect_views(res)
    path = out_dir / "python.xlsx"
    excel.write_workbook(path, views, labels, res.alpha)
    return _sheets(path), views, list(labels)


def _references(fx, views):
    """(Python sheet prefix, Python view, MATLAB struct holding the same effect), in the order
    Python writes them."""
    def items(x):
        """A MATLAB struct-array field as a list; empty fields load as empty arrays."""
        return pu.as_list(x) if np.size(x) else []

    res = fx["results"]
    ref = {"main": res}
    for e in items(res.get("effects", [])):
        ref[e["key"]] = e
        ref.update({s["key"]: s for s in items(e.get("simple_effects", []))})
    out = []
    for v in views:
        parent = ref[v.key]
        children = {p["label"]: p for p in items(parent.get("posthoc", []))}
        out += [(v, parent)] + [(p, children[p.title]) for p in v.posthoc]
    return [("" if i == 0 else f"{i + 1}_", v, r) for i, (v, r) in enumerate(out)]


def _matlab_round(x):
    """MATLAB's round(x, 4): half away from zero."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) * 1e4 + 0.5) / 1e4


def _assert_cells(actual, expected, unrounded, what):
    """Rounded numeric cells equal (None = empty cell = NaN). On a mismatch the message gives
    MATLAB's unrounded value, so a rounding-boundary case can be told from a real difference."""
    a = np.array([np.nan if c is None else c for c in actual], dtype=float)
    e = np.array([np.nan if c is None else c for c in expected], dtype=float)
    assert a.shape == e.shape, f"{what}: {a.size} cells vs {e.size}"
    assert np.array_equal(np.isnan(a), np.isnan(e)), f"{what}: empty cells differ"
    bad = np.flatnonzero(np.abs(np.nan_to_num(a - e)) > 1e-12)
    u = np.asarray(unrounded, dtype=float).ravel()
    assert bad.size == 0, "; ".join(f"{what}: channel {i + 1} {a[i]} vs {e[i]} (MATLAB "
                                    f"unrounded {u[i]!r})" for i in bad[:5])


def _sig_list(rows, matlab_layout):
    """1-based channel indices from either layout of a significant-channel sheet."""
    if len(rows) == 2 and rows[1] == [NO_SIG]:
        return []
    if matlab_layout:
        assert len(rows) == 2, "MATLAB writes the indices as one row"
        return [int(v) for v in rows[1]]
    assert rows[0] == ["ChannelIndex", "ChannelLabel"]
    return [int(r[0]) for r in rows[1:]]


def _check_sig(py_rows, expected, labels, name, what):
    got = _sig_list(py_rows, matlab_layout=False)
    if not (len(py_rows) == 2 and py_rows[1] == [NO_SIG]):
        assert [r[1] for r in py_rows[1:]] == [labels[i - 1] for i in got], f"{what} labels"
    if name == "correctSnPMsigch":
        assert sorted(got) == sorted(expected), what
    else:
        assert got == expected, what


def _ref_index(r, field):
    return [] if not np.size(r[field]) else (pu.index0(r[field]) + 1).tolist()


@pytest.mark.parametrize("case", CASES)
def test_workbook_matches_matlab(case, tmp_path):
    fx = _load(case)
    ml = _sheets(WORKBOOKS / f"{case}.xlsx")
    py, views, labels = _python_workbook(fx, tmp_path)
    refs = _references(fx, views)

    # ---- the effect MATLAB's workbook holds: main, or the mixed2way interaction ----
    target = "interaction" if "effects" in fx["results"] else "main"
    prefix = next(pre for pre, v, _ in refs if v.key == target)
    res = fx["results"]
    ours = {n[len(prefix):] for n in py if n.startswith(prefix)} if prefix else \
        {n for n in py if not n[0].isdigit() and n != "contents"}
    assert ours == set(ml) - {"posthoc"}, "sheet names"

    for name, (grp, field) in MAP_SHEETS.items():
        m, p = ml[name], py[prefix + name]
        assert len(m) == len(p) == 2 and len(m[0]) == len(p[0]) == len(labels), name
        assert p[0] == labels, f"{name} header"
        assert m[0] == labels, f"{name} MATLAB header"
        _assert_cells(p[1], m[1], res[grp][field], f"{case} {name}")

    if "effectiveN" in ml:
        m, p = ml["effectiveN"], py[prefix + "effectiveN"]
        assert m == p, "effectiveN"

    m, p = ml["excludedChannels"], py[prefix + "excludedChannels"]
    if m[0] == ["ChannelLabel", "Reason"]:
        assert p[0] == ["ChannelLabel", "Reason"] and len(m) == len(p) == 2
        assert p[1] == [m[1][0], m[1][1]], "excludedChannels (none)"
    else:
        assert m[0] == ["ChannelIndex", "ChannelLabel", "Reason", "Analysis"]
        assert p[0] == ["ChannelIndex", "ChannelLabel", "Reason"]
        assert [r[:2] for r in p[1:]] == [r[:2] for r in m[1:]], "excluded channels"
        assert [r[2] for r in p[1:]] == [r[2] for r in m[1:]], "reasons"

    for name in SIG_SHEETS:
        _check_sig(py[prefix + name], _sig_list(ml[name], matlab_layout=True), labels, name,
                   f"{case} {name}")

    # ---- MATLAB's post-hoc summary against Python's contents and prefixed sheets ----
    if "posthoc" in ml:
        head, *rows = ml["posthoc"]
        assert head == ["Effect", "Pair", "Status", "RunOn", "Rows", "Subjects",
                        "SubjectsLeftOut", "TFCEsigChannels", "ClusterSigChannels"]
        contents = {r[1]: dict(zip(py["contents"][0], r)) for r in py["contents"][1:]}
        for eff, pair, status, run_on, _, _, left_out, n_tfce, n_cluster in rows:
            c = contents[pair]
            assert (c["Status"], c["RunOn"], c["SubjectsLeftOut"] or None) == \
                (status, run_on, left_out), f"posthoc {pair}"
            assert any(v.title.endswith(eff) and any(q.title == pair for q in v.posthoc)
                       for v in views), f"posthoc {pair} under {eff}"
            pre = c["Prefix"] or ""
            assert len(_sig_list(py[pre + "correctTFCEsigch"], False)) == n_tfce, pair
            assert len(_sig_list(py[pre + "correctSnPMsigch"], False)) == n_cluster, pair
    else:
        assert not any(v.posthoc or v.skipped for v in views)

    # ---- every other Python effect against the maps MATLAB kept in its results ----
    for pre, v, r in refs:
        if pre == prefix:
            continue
        for name, (grp, field) in MAP_SHEETS.items():
            p = py[pre + name]
            assert p[0] == labels, f"{pre}{name} header"
            exp = [None if np.isnan(x) else x for x in _matlab_round(r[grp][field])]
            _assert_cells(p[1], exp, r[grp][field], f"{case} {v.key} {name}")
        for name, field in SIG_SHEETS.items():
            _check_sig(py[pre + name], _ref_index(r, field), labels, name,
                       f"{case} {v.key} {name}")

