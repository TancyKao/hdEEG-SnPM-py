"""Channels with no variance (Method Spec of 2026-10-14, `_scratch/glm_ruling/zero_variance_spec.md`).

Rule 1: a channel with nothing to permute is excluded once, before the loop, with a reason.
Rule 2: at any labelling whose error sum of squares is zero (SS <= 1e-20 * sum(v ** 2)) the
statistic is 0. Rule 3: an observed channel that hit rule 2 is NaN in the observed map and listed
as excluded, but stays in every permuted map. The numbered tests are the spec's acceptance tests.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import glm, legacy, missing, pipeline
from hdeeg_snpm.output import run_and_write
from hdeeg_snpm.pipeline import prepare
from test_invariants import LABELS, N_CH, params


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def glm_table(folder, meta, Y, name="glm"):
    frame = pd.concat([meta.reset_index(drop=True),
                       pd.DataFrame(Y, columns=LABELS[:Y.shape[1]])], axis=1)
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def reasons_of(result):
    return dict(zip(result.excluded_channels.tolist(), result.excluded_reasons))


# --------------------------------------------------------------------------- GLM tier

def test_1_anova1_constant_channel_is_excluded_with_its_reason(egi_graph):
    rng = np.random.default_rng(1)
    meta = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 6)})
    Y = rng.standard_normal((18, N_CH))
    Y[:, 7] = 0.1
    loops = quiet(glm.run_preset, "anova1", Y, meta, egi_graph, n_perm=50, rng=rng,
                  group_col="group")
    for lp in loops:
        assert reasons_of(lp.result) == {7: glm.NO_VARIATION}, lp.role
        assert np.isnan(lp.result.real_T[7]) and np.isnan(lp.result.p_correctedTFCE[7])
        assert np.isfinite(np.delete(lp.result.real_T, 7)).all()


def test_2_ancova_pair_excludes_a_channel_its_omnibus_tested(egi_graph):
    rng = np.random.default_rng(2)
    meta = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 7), "age": rng.normal(40, 9, 21)})
    Y = rng.standard_normal((21, N_CH))
    Y[:14, 5] = 0.7                          # constant within a and b, varying in c
    loops = {lp.role: lp for lp in quiet(glm.run_preset, "ancova", Y, meta, egi_graph,
                                         n_perm=50, rng=rng, group_col="group",
                                         covariate_cols=["age"])}
    assert np.isfinite(loops["main"].result.real_T[5]) and 5 not in reasons_of(loops["main"].result)
    pair = loops["main/posthoc:a vs b"].result
    assert reasons_of(pair) == {5: glm.NO_VARIATION} and np.isnan(pair.real_T[5])
    for other in ("main/posthoc:a vs c", "main/posthoc:b vs c"):
        assert np.isfinite(loops[other].result.real_T[5]), other


def test_rmanova_channel_constant_within_every_subject_completes(egi_graph):
    """The 1-df rmanova case that once asked TFCE for 85 GB: nothing to permute."""
    rng = np.random.default_rng(3)
    meta = pd.DataFrame({"Subject": np.repeat(np.arange(6), 2), "cond": np.tile(["x", "y"], 6)})
    Y = rng.standard_normal((12, N_CH))
    Y[:, 9] = np.repeat(rng.integers(0, 3, 6), 2).astype(float)
    lp = quiet(glm.run_preset, "rmanova", Y, meta, egi_graph, n_perm=40, rng=rng,
               subject_col="Subject", condition_col="cond")[0]
    assert reasons_of(lp.result) == {9: glm.NO_VARIATION}


def test_glm_zero_error_term_gives_statistic_zero_and_observed_perfect_fit_is_reported(egi_graph):
    """Two groups whose values are constant within each group: observed F has no error term."""
    rng = np.random.default_rng(4)
    meta = pd.DataFrame({"group": np.repeat(["a", "b"], 6)})
    Y = rng.standard_normal((12, N_CH))
    Y[:, 11] = np.repeat([0.1, 0.3], 6)
    stat, p, _, _, und = glm.glm_stat(Y, glm.build_design("anova1", meta, group_col="group").X,
                                      glm.build_design("anova1", meta, group_col="group").C,
                                      return_undefined=True)
    assert und.tolist() == [i == 11 for i in range(N_CH)] and stat[11] == 0 and p[11] == 1
    lp = quiet(glm.run_preset, "anova1", Y, meta, egi_graph, n_perm=40, rng=rng,
               group_col="group")[0]
    assert reasons_of(lp.result) == {11: glm.PERFECT_FIT} and np.isnan(lp.result.real_T[11])
    # it stays in the null: the whole-head average still covers it
    assert lp.global_mask[11]


def test_mixed2way_simple_effect_needs_two_subjects_and_says_why_in_every_output(tmp_path):
    rng = np.random.default_rng(5)
    subj = [f"s{i:02d}" for i in range(7)]
    meta = pd.DataFrame({"Subject": np.repeat(subj, 2),
                         "group": np.repeat(["g0"] * 6 + ["g1"], 2),
                         "cond": np.tile(["pre", "post"], 7)})
    Y = rng.standard_normal((14, N_CH))
    path = glm_table(tmp_path, meta, Y)
    with pytest.warns(UserWarning, match="within g1 skipped"):
        paths = run_and_write({"comparison": "mixed2way", "channels": "egi", "data_file": path,
                               "meta_cols": ["Subject", "group", "cond"], "group_col": "group",
                               "condition_col": "cond", "subject_col": "Subject",
                               "permutations": 20, "seed": 1, "output_path": str(tmp_path)})
    summary = json.loads(paths["json"].read_text())
    keys = [e["key"] for e in summary["effects"]]
    assert "condition@g0" in keys and "condition@g1" not in keys
    inter = next(e for e in summary["effects"] if e["key"] == "interaction")
    assert [s["pair"] for s in inter["skipped_pairs"]] == ["simple effect: condition within g1"]
    assert "at least 2" in inter["skipped_pairs"][0]["reason"]
    assert "Simple effect: condition within g1 was not run" in paths["html"].read_text()
    contents = pd.read_excel(paths["xlsx"], sheet_name="contents")
    assert contents.Status.str.startswith("not run").sum() == 1


# --------------------------------------------------------------------------- t-test tier

def test_3_paired_differences_of_plus_minus_one_are_tested(egi_graph):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((8, N_CH))
    d = np.array([1, -1, 1, 1, -1, 1, -1, 1], float)
    y = x.copy() - rng.standard_normal((8, N_CH))
    y[:, 3] = x[:, 3] - d
    # row 0 the observed labelling (required by the legacy b / N p), then the sign flip that
    # makes every difference +1 (a zero error term) and its mirror
    perms = np.vstack([np.ones(8, bool), d > 0, d < 0, rng.random((60, 8)) > 0.5])
    T = legacy.single_threshold_tfce(x, y, egi_graph, "pairedT", perms=perms)
    C = legacy.cluster_analysis(x, y, 2.36, egi_graph, "pairedT", perms=perms)
    assert np.isfinite(T.real_T).all() and np.isfinite(T.tMaxTFCE).all()
    assert np.isfinite(C.real_stat).all()
    assert legacy.paired_t(np.where((d > 0)[:, None], x - y, y - x))[0][3] == 0


def test_3_paired_differences_all_plus_one_are_reported_as_perfect_separation(tmp_path):
    rng = np.random.default_rng(7)
    x = rng.standard_normal((8, N_CH))
    y = x - rng.standard_normal((8, N_CH))
    y[:, 3] = x[:, 3] - 1.0
    with pytest.warns(UserWarning, match="perfect separation"):
        paths = run_and_write({**params(tmp_path, "pairedT", x, y), "permutations": 60,
                               "seed": 2, "output_path": str(tmp_path)})
    effect = json.loads(paths["json"].read_text())["effects"][0]
    assert effect["excluded_channels"] == [LABELS[3]]
    assert effect["excluded_reasons"] == [missing.PERFECT_SEPARATION]
    assert np.isnan(np.load(paths["npz"])["main/stat"][3])
    sheet = pd.read_excel(paths["xlsx"], sheet_name="excludedChannels")
    assert sheet.ChannelLabel.tolist() == [LABELS[3]]
    assert sheet.Reason.tolist() == [missing.PERFECT_SEPARATION]
    page = paths["html"].read_text()
    assert "Perfect separation, t undefined (1): " + LABELS[3] in page


def test_4_unpaired_constant_groups_that_differ_are_perfect_separation(tmp_path, egi_graph):
    rng = np.random.default_rng(8)
    x, y = rng.standard_normal((6, N_CH)), rng.standard_normal((6, N_CH))
    x[:, 12], y[:, 12] = 0.1, 0.3
    prep = quiet(prepare, params(tmp_path, "unpairedT", x, y))
    assert prep.rule.evaluable[12] and prep.undefined.tolist() == [i == 12 for i in range(N_CH)]
    assert prep.excluded[0].tolist() == [12]
    assert prep.excluded[1] == [missing.PERFECT_SEPARATION]
    # the identity relabelling (zero error term) is in the null, the run completes
    perms = np.vstack([np.arange(12), np.r_[6:12, 0:6],
                       [rng.permutation(12) for _ in range(60)]])
    res = quiet(pipeline.run_legacy, {**params(tmp_path, "unpairedT", x, y), "permutations": 62},
                perms_tfce=perms, perms_cluster=perms)
    assert np.isnan(res.T.real_T[12]) and np.isnan(res.T.p_correctedTFCE[12])
    assert np.isfinite(np.delete(res.T.real_T, 12)).all() and np.isfinite(res.T.tMaxTFCE).all()


@pytest.mark.parametrize("comparison,reason", [
    ("pairedT", missing.NO_DIFFERENCE), ("onesampleT", missing.ALL_ZERO),
    ("unpairedT", missing.CONSTANT_POOLED)])
def test_t_test_channel_with_nothing_to_permute_is_excluded_up_front(tmp_path, comparison, reason):
    rng = np.random.default_rng(9)
    x, y = rng.standard_normal((10, N_CH)), rng.standard_normal((10, N_CH))
    x[:, 4] = y[:, 4] = 0.25 if comparison != "unpairedT" else 0.0
    prep = quiet(prepare, params(tmp_path, comparison, x, y))
    assert prep.rule.excluded.tolist() == [4] and prep.rule.reasons == [reason]
    assert np.isnan(prep.x[:, 4]).all()


def test_onesample_against_zero_with_all_zero_channel(tmp_path):
    rng = np.random.default_rng(10)
    x = rng.standard_normal((10, N_CH))
    x[:, 2] = 0.0
    p = params(tmp_path, "onesampleT", x, x)
    p.pop("data2_file")
    prep = quiet(prepare, p)
    assert prep.rule.excluded.tolist() == [2] and prep.rule.reasons == [missing.ALL_ZERO]


# --------------------------------------------------------------------------- correlation tier

@pytest.mark.parametrize("comparison", ["correlationP", "correlationS"])
def test_6_constant_correlation_columns_are_listed(tmp_path, comparison):
    rng = np.random.default_rng(11)
    x, y = rng.standard_normal((12, N_CH)), rng.standard_normal((12, N_CH))
    x[:, 6], y[:, 8] = 0.1, 0.0
    res = quiet(pipeline.run_legacy, {**params(tmp_path, comparison, x, y), "permutations": 30,
                                      "seed": 3})
    idx, why = res.prepared.excluded
    assert idx.tolist() == [6, 8] and why == [missing.CONSTANT] * 2
    assert np.isnan(res.T.real_T[[6, 8]]).all()
    assert np.isfinite(np.delete(res.T.real_T, [6, 8])).all()


@pytest.mark.parametrize("comparison", ["correlationP", "correlationS"])
def test_6_channel_that_is_a_linear_function_of_the_covariate_is_listed(tmp_path, comparison):
    rng = np.random.default_rng(12)
    x, y = rng.standard_normal((12, N_CH)), rng.standard_normal((12, N_CH))
    age = rng.normal(40, 9, 12)
    x[:, 15] = 2 * age + 1
    cov = pd.DataFrame({"Subject": [f"sub{i:03d}" for i in range(1, 13)], "age": age})
    cov.to_csv(tmp_path / "cov.csv", index=False)
    prep = quiet(prepare, params(tmp_path, comparison, x, y, use_covariates=True,
                                 covariate_file=str(tmp_path / "cov.csv")))
    assert prep.rule.excluded.tolist() == [15] and prep.rule.reasons == [missing.EXPLAINED]


def test_constant_behaviour_score_leaves_nothing_to_test(tmp_path):
    rng = np.random.default_rng(13)
    x = rng.standard_normal((10, N_CH))
    score = pd.DataFrame({"Subject": [f"sub{i:03d}" for i in range(1, 11)],
                          "score": 0.3 + 1e-14 * np.arange(10)})       # varies by rounding only
    score.to_csv(tmp_path / "score.csv", index=False)
    p = {**params(tmp_path, "correlationP", x, x), "data2_file": str(tmp_path / "score.csv")}
    with pytest.raises(ValueError, match="No channel can be tested"):
        quiet(prepare, p)


# --------------------------------------------------------------------------- no-op

def _disable(monkeypatch):
    for mod in (legacy, missing, glm):
        monkeypatch.setattr(mod, "ZERO_SS_RTOL", -np.inf)


def _legacy_arrays(res):
    return [res.T.real_T, res.T.real_TFCE, res.T.tMax, res.T.tMaxTFCE, res.T.p_corrected,
            res.T.p_correctedTFCE, res.T.p_real, res.clusters.max_cluster_sizes,
            np.array([c.p for c in res.clusters.clusters])]


@pytest.mark.parametrize("comparison", ["pairedT", "unpairedT", "onesampleT", "correlationP",
                                        "correlationS"])
def test_7_rule_is_a_no_op_on_complete_continuous_data(tmp_path, monkeypatch, comparison):
    rng = np.random.default_rng(14)
    x, y = rng.standard_normal((10, N_CH)), rng.standard_normal((10, N_CH)) + 0.3
    p = {**params(tmp_path, comparison, x, y), "permutations": 40, "seed": 4}
    on = quiet(pipeline.run_legacy, p)
    _disable(monkeypatch)
    off = quiet(pipeline.run_legacy, p)
    assert not on.prepared.excluded[0].size
    for a, b in zip(_legacy_arrays(on), _legacy_arrays(off)):
        assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize("preset", ["anova1", "rmanova"])
def test_7_glm_rule_is_a_no_op_on_complete_continuous_data(egi_graph, monkeypatch, preset):
    rng = np.random.default_rng(15)
    meta = (pd.DataFrame({"group": np.repeat(["a", "b", "c"], 6)}) if preset == "anova1" else
            pd.DataFrame({"Subject": np.repeat(np.arange(6), 3), "cond": np.tile(list("xyz"), 6)}))
    cols = ({"group_col": "group"} if preset == "anova1" else
            {"subject_col": "Subject", "condition_col": "cond"})
    Y = rng.standard_normal((18, N_CH))

    def run():
        return quiet(glm.run_preset, preset, Y, meta, egi_graph, n_perm=30,
                     rng=np.random.default_rng(5), **cols)

    on = run()
    _disable(monkeypatch)
    off = run()
    for a, b in zip(on, off):
        assert a.result.excluded_channels.size == 0
        for f in ("real_T", "real_TFCE", "tMaxTFCE", "maxMass", "p_correctedTFCE", "p_real"):
            assert np.array_equal(getattr(a.result, f), getattr(b.result, f), equal_nan=True), f
        assert [c.p for c in a.result.clusters] == [c.p for c in b.result.clusters]


# --------------------------------------------------------------------------- round 4, question 3

def test_mixed2way_whole_head_perfect_fit_is_undefined_with_its_reason(tmp_path):
    """Channel average exactly additive in group and condition: every whole-head fit is perfect,
    so statistic and p are NaN with the reason, in every output; the maps are still tested."""
    rng = np.random.default_rng(16)
    subj = [f"s{i:02d}" for i in range(8)]
    meta = pd.DataFrame({"Subject": np.repeat(subj, 2), "group": np.repeat(["g0", "g1"], 8),
                         "cond": np.tile(["pre", "post"], 8)})
    noise = rng.standard_normal((16, N_CH))
    noise -= noise.mean(axis=1, keepdims=True)
    avg = (meta.group == "g1") * 0.5 + (meta.cond == "post") * 0.3 \
        + ((meta.group == "g1") & (meta.cond == "post")) * 0.2
    path = glm_table(tmp_path, meta, noise + avg.to_numpy(float)[:, None])
    paths = quiet(run_and_write, {
        "comparison": "mixed2way", "channels": "egi", "data_file": path,
        "meta_cols": ["Subject", "group", "cond"], "group_col": "group", "condition_col": "cond",
        "subject_col": "Subject", "permutations": 20, "seed": 1, "output_path": str(tmp_path)})
    effects = json.loads(paths["json"].read_text())["effects"]
    assert [e["key"] for e in effects] == ["group", "condition", "interaction", "condition@g0",
                                           "condition@g1"]
    for e in effects:
        assert e["global"]["reason"] == glm.PERFECT_FIT, e["key"]
        assert e["global"]["statistic"] is None or np.isnan(e["global"]["statistic"])
        assert e["global"]["p"] is None or np.isnan(e["global"]["p"])
    page = paths["html"].read_text()
    assert page.count(f"not computed: {glm.PERFECT_FIT}") == len(effects)
    contents = pd.read_excel(paths["xlsx"], sheet_name="contents")
    assert contents.WholeHead.str.endswith(glm.PERFECT_FIT).sum() == len(effects)
    assert np.isfinite(np.load(paths["npz"])["interaction/stat"]).all()


def test_whole_head_global_is_unchanged_when_the_fit_is_not_perfect(egi_graph):
    rng = np.random.default_rng(17)
    meta = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 6)})
    Y = rng.standard_normal((18, N_CH))
    lp = glm.plan_preset("anova1", Y, meta, group_col="group")[0]
    stat, p, df1, df2, why = lp.global_test()
    ref = glm.glm_stat(Y.mean(axis=1)[:, None], lp.design.X, lp.C)
    assert why == "" and (df1, df2) == (2, 15)
    np.testing.assert_allclose([stat, p], [ref[0][0], ref[1][0]], rtol=1e-12)


def test_rmanova_with_no_error_degrees_of_freedom_is_refused():
    """One subject with both conditions, two with one row each: 4 rows, rank 4."""
    meta = pd.DataFrame({"Subject": ["s1", "s1", "s2", "s3"], "cond": ["x", "y", "x", "y"]})
    Y = np.random.default_rng(18).standard_normal((4, N_CH))
    with pytest.raises(ValueError, match=glm.NO_ERROR_DF):
        glm.plan_preset("rmanova", Y, meta, subject_col="Subject", condition_col="cond")


def test_anova1_with_one_subject_per_group_is_refused(tmp_path):
    meta = pd.DataFrame({"Subject": ["s1", "s2", "s3"], "group": ["a", "b", "c"]})
    path = glm_table(tmp_path, meta, np.random.default_rng(19).standard_normal((3, N_CH)))
    with pytest.raises(ValueError, match=glm.NO_ERROR_DF):
        pipeline.run({"comparison": "anova1", "channels": "egi", "data_file": path,
                      "meta_cols": ["Subject", "group"], "group_col": "group",
                      "permutations": 20})


def test_mixed2way_with_one_subject_per_group_is_refused():
    meta = pd.DataFrame({"Subject": np.repeat(["s1", "s2"], 2), "group": np.repeat(["a", "b"], 2),
                         "cond": np.tile(["x", "y"], 2)})
    Y = np.random.default_rng(20).standard_normal((4, N_CH))
    with pytest.raises(ValueError, match=glm.NO_ERROR_DF):
        quiet(glm.plan_preset, "mixed2way", Y, meta, group_col="group", condition_col="cond",
              subject_col="Subject")
