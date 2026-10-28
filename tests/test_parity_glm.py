"""GLM tier against MATLAB: each permutation loop of each fixture is replayed from MATLAB's own
design, data and Freedman-Lane row indices, and must reproduce MATLAB's maps and nulls."""

import numpy as np
import pandas as pd
import pytest

import parity_util as pu
from hdeeg_snpm.glm import Design, plan_preset, run_effect, run_preset

pytestmark = pytest.mark.parity

CASES = ["anova1", "ancova_2group", "ancova_3group", "regression", "rmanova", "compu_anova1",
         "mixed2way_2x2", "mixed2way_2x3"]


def _replay(fx, k):
    loop = pu.as_list(fx["glm"]["loops"])[k]
    log = pu.as_list(fx["perm_log"])[k]
    X = np.atleast_2d(np.asarray(loop["X"], dtype=float))
    C = np.asarray(loop["C"], dtype=float)
    C = C.reshape(1, -1) if C.ndim == 1 else C
    design = Design(X=X, C=C, nuisance=np.asarray(loop["nuisance_idx"]).astype(bool),
                    blocks=np.asarray(loop["eb"]).astype(np.int64),
                    perm_type=loop["perm_type"], effect_label=loop["name"])
    out = run_effect(np.asarray(loop["Y"], dtype=float), design, pu.graph_of(fx),
                     perms=pu.index0(log["indices"]), alpha=float(fx["glm"]["alpha"]),
                     E=float(fx["glm"]["tfce_E"]), H=float(fx["glm"]["tfce_H"]),
                     cluster_statistic=pu.cluster_statistic(fx))
    return loop, out


@pytest.mark.parametrize("case", CASES)
def test_every_loop_matches_matlab(case):
    fx = pu.load(case)
    loops = pu.as_list(fx["glm"]["loops"])
    assert len(loops) == len(pu.as_list(fx["perm_log"]))
    for k, _ in enumerate(loops):
        loop, out = _replay(fx, k)
        what = f"{case} loop {k + 1} ({loop['name']})"
        assert loop["replay_ok"], what
        two_sided = loop["contrast_type"] == "t"
        assert np.array_equal(~np.isnan(out.real_T), np.asarray(loop["evaluable"]).astype(bool))
        pu.assert_close(out.real_T, loop["stat"], pu.STAT_RTOL, f"{what} statistic")
        pu.assert_close(out.p_real, loop["p_uncorrected"], 1e-8, f"{what} uncorrected p")
        pu.assert_close(out.real_TFCE, loop["tfce"], pu.NULL_RTOL, f"{what} TFCE map")
        pu.assert_close(out.tMaxTFCE, loop["tfce_null_sorted"], pu.NULL_RTOL, f"{what} TFCE null")
        pu.assert_close(out.maxMass, loop["cluster_mass_null_sorted"], pu.NULL_RTOL,
                        f"{what} cluster-mass null")
        expected_p = pu.fwe_p(loop["tfce"], loop["tfce_null_sorted"], two_sided)
        ties = pu.assert_p_equal(out.p_correctedTFCE, expected_p, loop["tfce"],
                                 loop["tfce_null_sorted"], f"{what} TFCE p", two_sided)
        assert not ties, f"{what}: p differs by one count at channels {ties}"


@pytest.mark.parametrize("case", ["anova1", "ancova_2group", "ancova_3group", "regression",
                                  "rmanova", "compu_anova1"])
def test_main_effect_matches_pipeline_results(case):
    """The first loop is the main effect; compare with what the MATLAB pipeline itself reported."""
    fx = pu.load(case)
    _, out = _replay(fx, 0)
    res = fx["results"]
    assert np.array_equal(out.p_correctedTFCE, np.asarray(res["p"]["correctedTFCE"], dtype=float),
                          equal_nan=True)
    _assert_same_clusters(out.clusters, res["Clusters"], case)


def _assert_same_clusters(ours, theirs, what):
    """Same channel sets, identical corrected p, cluster mass within the null tolerance."""
    theirs = [c for c in pu.as_list(theirs) if np.size(c["channels"])]
    ours = [c for c in ours if c.channels.size]
    assert len(ours) == len(theirs), what
    key = {frozenset(pu.index0(c["channels"]).tolist()): c for c in theirs}
    for c in ours:
        ref = key[frozenset(c.channels.tolist())]
        assert c.p == float(ref["p"]), what
        assert abs(c.mass - float(ref["mass"])) <= pu.NULL_RTOL * abs(float(ref["mass"])), what


def _table(fx):
    d = fx["inputs"]["data"]
    meta = pd.DataFrame({k: np.asarray(d["meta"][k]) for k in np.atleast_1d(d["meta_names"])})
    for k in meta:
        if meta[k].dtype.kind in "ui":
            meta[k] = meta[k].astype(float)
    Y = np.asarray(d["values"], dtype=float)[:, pu.index0(fx["montage"]["data_columns"])]
    p = fx["params"]
    cols = {k: (list(np.atleast_1d(p[k])) if k == "covariate_cols" else p[k])
            for k in ("group_col", "condition_col", "subject_col", "predictor_col",
                      "covariate_cols") if k in p and np.size(p[k])}
    return Y, meta, cols


@pytest.mark.parametrize("case", CASES)
def test_plan_built_from_the_input_table_matches_matlab(case):
    """Designs, contrasts, row sets and loop order built from the raw table, not taken from MATLAB."""
    fx = pu.load(case)
    Y, meta, cols = _table(fx)
    plan = plan_preset(fx["params"]["comparison"], Y, meta, **cols)
    ref = pu.as_list(fx["glm"]["loops"])
    assert [lp.role for lp in plan] == [r["role"] for r in ref]
    current = pu.matlab_has_2026_10_14_glm_fix(fx)
    for lp, r in zip(plan, ref):
        what = f"{case} {lp.role}"
        if not current and pu.changed_2026_10_14(lp.role):
            continue
        assert np.array_equal(lp.rows, pu.index0(r["rows"])), what
        pu.assert_close(lp.Y, r["Y"], 1e-14, f"{what} Y")
        pu.assert_close(lp.design.X, np.atleast_2d(r["X"]), 1e-14, f"{what} X")
        pu.assert_close(lp.C, r["C"], 0, f"{what} C")
        assert np.array_equal(lp.design.nuisance, np.asarray(r["nuisance_idx"]).astype(bool)), what
        assert lp.design.perm_type == r["perm_type"], what
        # block ids may be numbered differently; the partition must be the same
        assert len(set(zip(lp.design.blocks.tolist(), np.asarray(r["eb"]).tolist()))) == \
            len(set(lp.design.blocks.tolist())) == len(set(np.asarray(r["eb"]).tolist())), what


@pytest.mark.parametrize("case", ["mixed2way_2x2", "mixed2way_2x3", "ancova_3group"])
def test_preset_end_to_end_from_the_input_table(case):
    fx = pu.load(case)
    Y, meta, cols = _table(fx)
    if not pu.matlab_has_2026_10_14_glm_fix(fx):
        pytest.skip("reference file predates the 2026-10-14 post-hoc and mixed2way changes")
    perms = [pu.index0(lg["indices"]) for lg in pu.as_list(fx["perm_log"])]
    loops = run_preset(fx["params"]["comparison"], Y, meta, pu.graph_of(fx), perms=perms,
                       cluster_statistic=pu.cluster_statistic(fx), **cols)
    for lp, r in zip(loops, pu.as_list(fx["glm"]["loops"])):
        what = f"{case} {lp.role}"
        pu.assert_close(lp.result.real_T, r["stat"], pu.STAT_RTOL, f"{what} statistic")
        pu.assert_close(lp.result.tMaxTFCE, r["tfce_null_sorted"], pu.NULL_RTOL, f"{what} null")
        pu.assert_close(lp.result.maxMass, r["cluster_mass_null_sorted"], pu.NULL_RTOL,
                        f"{what} mass null")


def _glm_params(fx):
    p = fx["params"]
    out = {k: p[k] for k in ("comparison", "channels", "datatype", "group_col", "condition_col",
                              "subject_col", "predictor_col") if k in p and np.size(p[k])}
    out["meta_cols"] = list(np.atleast_1d(p["meta_cols"]))
    if "covariate_cols" in p and np.size(p["covariate_cols"]):
        out["covariate_cols"] = list(np.atleast_1d(p["covariate_cols"]))
    out["data_file"] = fx["source_files"]["data_file"]["absolute"]
    out["cluster_statistic"] = pu.cluster_statistic(fx)
    return out


@pytest.mark.parametrize("case", CASES)
def test_pipeline_from_the_file_matches_matlab_results(case):
    """Whole pipeline from the CSV file, compared with what MATLAB reported to the user."""
    from pathlib import Path

    from hdeeg_snpm.pipeline import run_glm

    fx = pu.load(case)
    params = _glm_params(fx)
    if not Path(params["data_file"]).exists():
        pytest.skip("input file is not on this machine")
    has_posthoc = any("/posthoc:" in r["role"] for r in pu.as_list(fx["glm"]["loops"]))
    if not pu.matlab_has_2026_10_14_glm_fix(fx) and (has_posthoc or case.startswith("mixed2way")):
        pytest.skip("reference file predates the 2026-10-14 post-hoc and mixed2way changes")
    perms = [pu.index0(lg["indices"]) for lg in pu.as_list(fx["perm_log"])]
    out = run_glm(params, perms=perms)
    res = fx["results"]
    assert list(out.setup.labels) == list(fx["montage"]["labels"])
    if case.startswith("mixed2way"):
        for eff in pu.as_list(res["effects"]):
            lp = out.loop(eff["key"])
            pu.assert_close(lp.result.real_T, eff["T"]["real_T"], pu.STAT_RTOL, eff["key"])
            assert np.array_equal(lp.result.p_correctedTFCE,
                                  np.asarray(eff["p"]["correctedTFCE"], float), equal_nan=True)
            _assert_same_clusters(lp.result.clusters, eff["Clusters"], eff["key"])
            stat, pval, df1, df2, _ = lp.global_test()
            pu.assert_close(stat, eff["global_stat"], 1e-10, f"{eff['key']} whole-head statistic")
            pu.assert_close(pval, eff["global_pval"], 1e-8, f"{eff['key']} whole-head p")
            assert (df1, df2) == (int(eff["global_df1"]), int(eff["global_df2"]))
        for simple in pu.as_list(pu.as_list(res["effects"])[2]["simple_effects"]):
            lp = out.loop(simple["key"])
            assert np.array_equal(lp.result.p_correctedTFCE,
                                  np.asarray(simple["p"]["correctedTFCE"], float), equal_nan=True)
            _assert_same_clusters(lp.result.clusters, simple["Clusters"], simple["key"])
    else:
        main = out.loop("main")
        pu.assert_close(main.result.real_T, res["T"]["real_T"], pu.STAT_RTOL, "statistic")
        assert np.array_equal(main.result.p_correctedTFCE,
                              np.asarray(res["p"]["correctedTFCE"], float), equal_nan=True)
        posthoc = [lp for lp in out.loops if "/posthoc:" in lp.role]
        ref = pu.as_list(res["posthoc"]) if np.size(res.get("posthoc", [])) else []
        ref = [r for r in ref if np.size(r.get("label", ""))]
        assert [lp.role.split("posthoc:")[1] for lp in posthoc] == [r["label"] for r in ref]
        for lp, r in zip(posthoc, ref):
            assert np.array_equal(lp.result.p_correctedTFCE,
                                  np.asarray(r["p"]["correctedTFCE"], float), equal_nan=True)
