"""GLM tier: statistics against independent computations, planted effects, confound control."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from conftest import DATA
from hdeeg_snpm import glm


def cluster_of(graph, centre=23):
    return sorted({centre, *graph.neighbor_lists[centre]})


def found(cluster, p_values, misses=1):
    """The planted channels are significant, allowing ``misses`` edge channels to fall short."""
    sig = set(np.flatnonzero(p_values < 0.05).tolist())
    return len(set(cluster) - sig) <= misses and len(sig - set(cluster)) <= 4


def f_by_model_comparison(Y, X_full, X_reduced):
    """F from the drop in residual sum of squares, computed without the contrast machinery."""
    def rss(X):
        return ((Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]) ** 2).sum(axis=0)
    df1 = X_full.shape[1] - X_reduced.shape[1]
    df2 = Y.shape[0] - X_full.shape[1]
    return (rss(X_reduced) - rss(X_full)) / df1 / (rss(X_full) / df2)


def test_anova1_matches_scipy(egi_graph):
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"group": np.repeat(["b", "a", "c"], [8, 9, 7])})
    Y = rng.standard_normal((24, egi_graph.n))
    d = glm.build_design("anova1", meta, group_col="group")
    stat, p, df1, df2 = glm.glm_stat(Y, d.X, d.C)
    ref = stats.f_oneway(*(Y[meta.group == g] for g in "bac"))
    np.testing.assert_allclose(stat, ref.statistic, rtol=1e-9)
    np.testing.assert_allclose(p, ref.pvalue, rtol=1e-8)
    assert (df1, df2) == (2, 21) and d.contrast_type == "F" and d.perm_type == "free"
    assert d.posthoc == [("b vs a", "b", "a"), ("b vs c", "b", "c"), ("a vs c", "a", "c")]
    assert d.factor_col == "group"


def test_regression_and_ancova_match_independent_fits(egi_graph):
    rng = np.random.default_rng(1)
    n = 30
    meta = pd.DataFrame({"group": np.tile(["x", "y", "z"], 10), "age": rng.normal(40, 10, n),
                         "sex": rng.choice(["M", "F"], n), "score": rng.standard_normal(n)})
    Y = rng.standard_normal((n, egi_graph.n))

    d = glm.build_design("regression", meta, predictor_col="score")
    t, p, _, _ = glm.glm_stat(Y, d.X, d.C)
    ref = [stats.linregress(meta.score, Y[:, j]) for j in range(5)]
    np.testing.assert_allclose(t[:5], [r.slope / r.stderr for r in ref], rtol=1e-9)
    np.testing.assert_allclose(p[:5], [r.pvalue for r in ref], rtol=1e-8)

    d = glm.build_design("ancova", meta, group_col="group", covariate_cols=["age", "sex"])
    assert d.X.shape == (n, 5) and d.nuisance.tolist() == [True, False, False, True, True]
    F = glm.glm_stat(Y, d.X, d.C)[0]
    np.testing.assert_allclose(F, f_by_model_comparison(Y, d.X, d.X[:, d.nuisance]), rtol=1e-8)
    assert [lab for lab, _, _ in d.posthoc] == ["x vs y", "x vs z", "y vs z"]


def test_rmanova_matches_model_comparison_and_blocks(egi_graph):
    rng = np.random.default_rng(2)
    subj = np.repeat(np.arange(10), 3)
    meta = pd.DataFrame({"Subject": subj, "condition": np.tile(["pre", "mid", "post"], 10)})
    Y = rng.standard_normal((30, egi_graph.n)) + rng.standard_normal(10)[subj, None] * 2
    d = glm.build_design("rmanova", meta, subject_col="Subject", condition_col="condition")
    assert d.perm_type == "within" and (d.blocks == subj).all()
    F = glm.glm_stat(Y, d.X, d.C)[0]
    np.testing.assert_allclose(F, f_by_model_comparison(Y, d.X, d.X[:, d.nuisance]), rtol=1e-8)


def test_partial_and_all_nan_channels(egi_graph):
    rng = np.random.default_rng(3)
    meta = pd.DataFrame({"group": np.repeat(["a", "b"], 10)})
    Y = rng.standard_normal((20, egi_graph.n))
    Y[3, 0] = np.nan
    Y[:, 1] = np.nan
    d = glm.build_design("anova1", meta, group_col="group")
    t = glm.glm_stat(Y, d.X, d.C)[0]
    keep = np.arange(20) != 3
    ref = stats.ttest_ind(Y[keep & (meta.group == "b"), 0], Y[keep & (meta.group == "a"), 0])
    np.testing.assert_allclose(t[0], ref.statistic, rtol=1e-9)
    assert np.isnan(t[1])
    ctx = glm.fl_context(Y, d)
    assert not ctx.evaluable[0] and not ctx.evaluable[1] and ctx.evaluable[2:].all()
    res = glm.run_effect(Y, d, egi_graph, n_perm=20, rng=np.random.default_rng(0))
    assert np.isnan(res.real_T[:2]).all() and np.isnan(res.p_correctedTFCE[:2]).all()
    assert res.excluded_channels.tolist() == [0, 1]


@pytest.mark.parametrize("preset", ["anova1", "rmanova"])
def test_planted_cluster_recovered(egi_graph, preset):
    rng = np.random.default_rng(4)
    cl = cluster_of(egi_graph)
    if preset == "anova1":
        meta = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 12)})
        Y = rng.standard_normal((36, egi_graph.n))
        Y[np.ix_(np.flatnonzero(meta.group == "c"), cl)] += 2.2
        d = glm.build_design(preset, meta, group_col="group")
    else:
        subj = np.repeat(np.arange(14), 3)
        meta = pd.DataFrame({"Subject": subj, "cond": np.tile(["a", "b", "c"], 14)})
        Y = rng.standard_normal((42, egi_graph.n)) + rng.standard_normal(14)[subj, None] * 3
        Y[np.ix_(np.flatnonzero(meta.cond == "c"), cl)] += 2.2
        d = glm.build_design(preset, meta, subject_col="Subject", condition_col="cond")
    res = glm.run_effect(Y, d, egi_graph, n_perm=300, rng=rng)
    assert found(cl, res.p_correctedTFCE)
    best = min(res.clusters, key=lambda c: c.p)
    assert best.p < 0.05 and len(set(cl) - set(best.channels.tolist())) <= 1
    # the planted level drives the post-hoc: a vs c significant, a vs b not
    cols = ({"group_col": "group"} if preset == "anova1"
            else {"subject_col": "Subject", "condition_col": "cond"})
    ph = {lp.role.split("posthoc:")[-1]: lp for lp in glm.plan_preset(preset, Y, meta, **cols)}
    hit = glm.run_effect(ph["a vs c"].Y, ph["a vs c"].design, egi_graph, n_perm=200,
                         C=ph["a vs c"].C, rng=rng)
    miss = glm.run_effect(ph["a vs b"].Y, ph["a vs b"].design, egi_graph, n_perm=200,
                          C=ph["a vs b"].C, rng=rng)
    assert found(cl, hit.p_correctedTFCE)
    assert (hit.real_T[cl] < 0).all() and np.nanmin(miss.p_correctedTFCE) > 0.05


def test_freedman_lane_controls_a_confound(egi_graph):
    """Groups differ in age and the signal depends on age only. Adjusting for age removes the
    apparent group effect; leaving it out does not."""
    rng = np.random.default_rng(5)
    n = 40
    group = np.repeat(["a", "b"], n // 2)
    age = np.where(group == "a", 30.0, 50.0) + rng.normal(0, 8, n)
    meta = pd.DataFrame({"group": group, "age": age})
    Y = rng.standard_normal((n, egi_graph.n)) + 0.12 * (age[:, None] - 40)
    adj = glm.run_effect(Y, glm.build_design("ancova", meta, group_col="group",
                                             covariate_cols=["age"]),
                         egi_graph, n_perm=200, rng=rng)
    raw = glm.run_effect(Y, glm.build_design("anova1", meta, group_col="group"),
                         egi_graph, n_perm=200, rng=rng)
    assert np.nanmin(adj.p_correctedTFCE) > 0.05
    assert (raw.p_correctedTFCE < 0.05).mean() > 0.5


def test_mixed2way_designs(egi_graph):
    subj = np.repeat(np.arange(12), 2)
    meta = pd.DataFrame({"Subject": subj, "group": np.repeat(["g1", "g2"], 12),
                         "cond": np.tile(["c1", "c2"], 12)})
    inter = glm.build_design("mixed2way", meta, group_col="group", subject_col="Subject",
                             condition_col="cond")
    cond = glm.build_design("mixed2way", meta, group_col="group", subject_col="Subject",
                            condition_col="cond", effect="condition")
    assert inter.X.shape == (24, 14) and inter.C.shape == (1, 14) and inter.C[0, -1] == 1
    # the condition effect is tested in the full model: same columns, interaction as nuisance
    assert cond.X.shape == (24, 14) and cond.perm_type == inter.perm_type == "within"
    assert cond.nuisance.sum() == 13 and not cond.nuisance[12] and cond.nuisance[13]
    assert set(np.unique(cond.X[:, 13])) == {-1.0, 0.0, 1.0}      # sum-to-zero group coding
    with pytest.raises(ValueError, match="between-subject"):
        glm.build_design("mixed2way", meta, group_col="group", subject_col="Subject",
                         condition_col="cond", effect="group")
    # a crossover interaction is found by the interaction contrast and not by the condition one
    rng = np.random.default_rng(6)
    cl = cluster_of(egi_graph)
    Y = rng.standard_normal((24, egi_graph.n)) + rng.standard_normal(12)[subj, None] * 2
    sign = np.where(meta.group == "g1", 1.0, -1.0) * np.where(meta.cond == "c2", 1.0, -1.0)
    Y[:, cl] += 1.6 * sign[:, None]
    ri = glm.run_effect(Y, inter, egi_graph, n_perm=300, rng=rng)
    rc = glm.run_effect(Y, cond, egi_graph, n_perm=300, rng=rng)
    assert found(cl, ri.p_correctedTFCE)
    assert np.nanmin(rc.p_correctedTFCE) > 0.05


def test_a_missing_design_value_is_an_error_that_names_the_column():
    import pandas as pd
    import pytest

    from hdeeg_snpm.glm import build_design
    meta = pd.DataFrame({"group": ["a", "b"] * 6, "sex": ["F", "M", None] * 4,
                         "age": [30.0, np.nan] + [40.0] * 10})
    with pytest.raises(ValueError, match="'sex'.*row"):
        build_design("ancova", meta, group_col="group", covariate_cols=["sex"])
    with pytest.raises(ValueError, match="'age'.*row"):
        build_design("regression", meta, predictor_col="age")


# --------------------------------------------------------------------------- post-hoc and mixed2way rules

def _three_groups(rng, n_ch, n=10):
    meta = pd.DataFrame({"group": np.tile(["a", "b", "c"], n),
                         "age": rng.normal(40, 10, 3 * n)})
    return meta, rng.standard_normal((3 * n, n_ch))


def test_posthoc_is_the_two_level_test_on_the_rows_of_the_two_levels(egi_graph):
    rng = np.random.default_rng(11)
    meta, Y = _three_groups(rng, egi_graph.n)
    Y[meta.group == "c"] += 5.0                          # the third level must not matter
    loops = {lp.role: lp for lp in glm.plan_preset("anova1", Y, meta, group_col="group")}
    lp = loops["main/posthoc:a vs b"]
    assert lp.Y.shape[0] == 20 and set(meta.group[lp.rows]) == {"a", "b"}
    a, b = Y[meta.group == "a"], Y[meta.group == "b"]
    ref = stats.ttest_ind(a, b)                           # mean(a) - mean(b), pooled over a and b only
    np.testing.assert_allclose(glm.glm_stat(lp.Y, lp.design.X, lp.C)[0], ref.statistic, rtol=1e-9)
    # identical to running the two-level preset on those rows with the same permutations
    two = meta[meta.group != "c"].reset_index(drop=True)
    alone = glm.build_design("anova1", two, group_col="group")
    perms = np.array([np.random.default_rng(i).permutation(20) for i in range(60)])
    r1 = glm.run_effect(lp.Y, lp.design, egi_graph, C=lp.C, perms=perms)
    r2 = glm.run_effect(Y[meta.group != "c"], alone, egi_graph, perms=perms)
    np.testing.assert_allclose(np.abs(r1.real_T), np.abs(r2.real_T), rtol=1e-10)
    assert np.array_equal(r1.p_correctedTFCE, r2.p_correctedTFCE)
    # b appears before c, a before c: sign is always first-named minus second-named
    lp = loops["main/posthoc:b vs c"]
    assert (glm.glm_stat(lp.Y, lp.design.X, lp.C)[0] < 0).mean() > 0.95


def test_within_subject_posthoc_is_the_paired_test(egi_graph):
    rng = np.random.default_rng(12)
    subj = np.repeat(np.arange(9), 3)
    meta = pd.DataFrame({"Subject": subj, "cond": np.tile(["a", "b", "c"], 9)})
    Y = rng.standard_normal((27, egi_graph.n)) + rng.standard_normal(9)[subj, None] * 2
    Y[meta.cond == "c"] *= 0.2                            # a quieter third level must not matter
    loops = {lp.role: lp for lp in glm.plan_preset("rmanova", Y, meta, subject_col="Subject",
                                                   condition_col="cond")}
    lp = loops["main/posthoc:a vs b"]
    ref = stats.ttest_rel(Y[meta.cond == "a"], Y[meta.cond == "b"])
    np.testing.assert_allclose(glm.glm_stat(lp.Y, lp.design.X, lp.C)[0], ref.statistic, rtol=1e-9)
    assert lp.design.perm_type == "within" and len(set(lp.design.blocks)) == 9


def test_posthoc_drops_subjects_with_one_of_the_two_levels_and_says_so(egi_graph):
    rng = np.random.default_rng(13)
    subj = np.repeat(np.arange(8), 3)
    meta = pd.DataFrame({"Subject": [f"s{i}" for i in subj], "cond": np.tile(["a", "b", "c"], 8)})
    Y = rng.standard_normal((24, egi_graph.n))
    keep = ~((meta.Subject == "s3") & (meta.cond == "b"))
    with pytest.warns(UserWarning, match="s3.*only one of the two"):
        loops = {lp.role: lp for lp in glm.plan_preset(
            "rmanova", Y[keep], meta[keep], subject_col="Subject", condition_col="cond")}
    assert loops["main/posthoc:a vs b"].Y.shape[0] == 14      # 7 subjects
    assert loops["main/posthoc:a vs c"].Y.shape[0] == 16      # all 8


def test_posthoc_that_cannot_be_estimated_is_skipped_with_a_warning(egi_graph):
    rng = np.random.default_rng(14)
    meta, Y = _three_groups(rng, egi_graph.n, n=6)
    meta["site"] = np.where(meta.group == "a", "x", "y")     # coincides with a versus the rest
    with pytest.warns(UserWarning, match="a vs b.*cannot be estimated"):
        plan = glm.plan_preset("ancova", Y, meta, group_col="group", covariate_cols=["site"])
    roles = [lp.role for lp in plan]
    assert "main/posthoc:a vs b" not in roles and "main/posthoc:b vs c" in roles
    assert [pair for pair, _ in plan[0].skipped] == ["a vs b", "a vs c"]


def test_posthoc_stays_nominal_when_a_third_level_differs_everywhere(egi_graph):
    """a = b, c raised at every channel. Permuting residuals that still carry c used to give
    the cluster test a false-positive rate of 0.2 to 0.5 here."""
    hits = np.zeros(2)
    n_sim = 60
    for seed in range(n_sim):
        rng = np.random.default_rng(1000 + seed)
        meta, Y = _three_groups(rng, egi_graph.n, n=12)
        Y[meta.group == "c"] += 3.0
        lp = next(x for x in glm.plan_preset("anova1", Y, meta, group_col="group")
                  if x.role == "main/posthoc:a vs b")
        r = glm.run_effect(lp.Y, lp.design, egi_graph, n_perm=99, C=lp.C, rng=rng)
        hits += [np.nanmin(r.p_correctedTFCE) <= 0.05, min(c.p for c in r.clusters) <= 0.05]
    assert (hits / n_sim <= 0.15).all(), hits / n_sim           # 0.05 +/- sampling error at 60


def _mixed(rng, n_ch, sizes=(8, 8), conds=("pre", "post")):
    ns = sum(sizes)
    subj = np.repeat(np.arange(ns), len(conds))
    group = np.repeat(np.repeat([f"g{k}" for k in range(len(sizes))], sizes), len(conds))
    meta = pd.DataFrame({"Subject": [f"s{i:02d}" for i in subj], "group": group,
                         "cond": np.tile(conds, ns)})
    return meta, rng.standard_normal((len(meta), n_ch)) + rng.standard_normal(ns)[subj, None]


MIXED = {"group_col": "group", "condition_col": "cond", "subject_col": "Subject"}


def test_mixed2way_needs_every_subject_to_have_every_condition(egi_graph):
    meta, Y = _mixed(np.random.default_rng(15), egi_graph.n)
    drop = ~((meta.Subject == "s05") & (meta.cond == "post"))
    with pytest.raises(ValueError, match=r"s05 \(post: 0 rows\)"):
        glm.plan_preset("mixed2way", Y[drop], meta[drop], **MIXED)
    twice = pd.concat([meta, meta[meta.Subject == "s02"].head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="s02"):
        glm.plan_preset("mixed2way", np.vstack([Y, Y[:1]]), twice, **MIXED)


def test_mixed2way_tests_every_effect_on_the_channels_present_in_every_row(egi_graph):
    rng = np.random.default_rng(16)
    meta, Y = _mixed(rng, egi_graph.n, conds=("pre", "mid", "post"))
    Y[4, 30] = np.nan                                        # one cell of one subject
    loops = glm.run_preset("mixed2way", Y, meta, egi_graph, n_perm=30, rng=rng, **MIXED)
    assert len(loops) == 1 + 4 + 1 + 4 + 4
    for lp in loops:
        assert lp.result.excluded_channels.tolist() == [30], lp.role


def test_mixed2way_condition_effect_is_the_standard_mixed_anova_F(egi_graph):
    """8 and 14 subjects, 3 conditions. MATLAB ranova and anovan (type III) both give
    F(2, 40) = 30.9004 for condition and 7.1276 for the interaction on this table."""
    wide = pd.read_csv(DATA / "mixed_anova_8_14_x3.csv")
    n = len(wide)
    meta = pd.DataFrame({"Subject": np.repeat(np.arange(n), 3),
                         "group": np.repeat(wide.group.to_numpy(), 3),
                         "cond": np.tile(["y1", "y2", "y3"], n)})
    y = wide[["y1", "y2", "y3"]].to_numpy().reshape(-1, 1)
    for effect, expected in (("condition", 30.9004), ("interaction", 7.1276)):
        d = glm.build_design("mixed2way", meta, effect=effect, **MIXED)
        F, _, df1, df2 = glm.glm_stat(y, d.X, d.C)
        assert (df1, df2) == (2, 40) and abs(F[0] - expected) < 1e-4, effect


def test_mixed2way_condition_effect_survives_a_crossover_interaction(egi_graph):
    """A real interaction used to stay in the permuted residuals and hide the condition effect."""
    rng = np.random.default_rng(17)
    meta, Y = _mixed(rng, egi_graph.n, sizes=(10, 10))
    cl = cluster_of(egi_graph)
    post = (meta.cond == "post").to_numpy()
    Y[post] += np.where(meta.group[post] == "g0", 3.0, -3.0)[:, None]     # crossover, everywhere
    Y[np.ix_(post, cl)] += 2.2                                             # condition effect
    loops = {lp.role: lp for lp in glm.plan_preset("mixed2way", Y, meta, **MIXED)}
    lp = loops["condition"]
    r = glm.run_effect(lp.Y, lp.design, egi_graph, n_perm=300, C=lp.C, rng=rng)
    assert found(cl, r.p_correctedTFCE)
    off = np.setdiff1d(np.arange(egi_graph.n), cl)
    assert (r.p_correctedTFCE[off] <= 0.05).mean() < 0.05


def test_skipped_pairs_and_left_out_subjects_reach_every_output_file(tmp_path):
    """A pair that is not run, or is run on fewer subjects, must be stated in the results file,
    the workbook and the report, not only in a console warning."""
    import json

    from hdeeg_snpm.montage import get_montage
    from hdeeg_snpm.output import run_and_write
    labels = list(get_montage("egi").labels)
    rng = np.random.default_rng(21)
    rows = [(f"sub{i:03d}", c) for i in range(1, 9) for c in ("C1", "C2", "C3")]
    rows.remove(("sub001", "C1"))
    rows.append(("sub002", "extra"))
    frame = pd.DataFrame(rng.standard_normal((len(rows), len(labels))), columns=labels)
    frame.insert(0, "condition", [c for _, c in rows])
    frame.insert(0, "Subject", [s for s, _ in rows])
    frame.to_csv(tmp_path / "t.csv", index=False)
    with pytest.warns(UserWarning):
        paths = run_and_write({"comparison": "rmanova", "channels": "egi",
                               "data_file": str(tmp_path / "t.csv"),
                               "meta_cols": ["Subject", "condition"], "subject_col": "Subject",
                               "condition_col": "condition", "permutations": 30, "seed": 1,
                               "output_path": str(tmp_path)})
    effect = json.loads(paths["json"].read_text())["effects"][0]
    # only sub002 has the "extra" condition, so none of its three pairs has 2 subjects
    assert {s["pair"] for s in effect["skipped_pairs"]} == {"C1 vs extra", "C2 vs extra",
                                                            "C3 vs extra"}
    pair = next(p for p in effect["posthoc"] if p["title"] == "C2 vs C1")   # sub001 comes first and has no C1
    assert pair["dropped_subjects"] == ["sub001"] and pair["run_on"].startswith("7 of 8")
    page = paths["html"].read_text()
    assert "C2 vs extra was not run" in page and "7 of 8 subjects" in page and "sub001" in page
    contents = pd.read_excel(paths["xlsx"], sheet_name="contents")
    assert contents.Status.str.startswith("not run").sum() == 3
    assert "sub001" in set(contents.SubjectsLeftOut.dropna())


def test_simple_effects_list_and_sign_their_pairs_as_the_whole_table_does(egi_graph):
    rng = np.random.default_rng(22)
    meta, Y = _mixed(rng, egi_graph.n, sizes=(6, 6), conds=("pre", "mid", "post"))
    # second group's rows arrive in a different condition order
    order = np.r_[np.arange(18), 18 + np.arange(18).reshape(6, 3)[:, ::-1].ravel()]
    meta, Y = meta.iloc[order].reset_index(drop=True), Y[order]
    plan = glm.plan_preset("mixed2way", Y, meta, **MIXED)
    roles = [lp.role for lp in plan]
    pairs = ["pre vs mid", "pre vs post", "mid vs post"]
    for scope in ("condition", "condition@g0", "condition@g1"):
        assert [r.split("posthoc:")[1] for r in roles if r.startswith(scope + "/")] == pairs
    # and the statistic is first-named minus second-named in both groups
    for g in ("g0", "g1"):
        for lp in (x for x in plan if x.role.startswith(f"condition@{g}/")):
            a, b = lp.role.split("posthoc:")[1].split(" vs ")
            rows = meta[meta.group == g].sort_values("Subject", kind="stable")
            ref = stats.ttest_rel(Y[rows.index[rows.cond == a]], Y[rows.index[rows.cond == b]])
            np.testing.assert_allclose(glm.glm_stat(lp.Y, lp.design.X, lp.C)[0], ref.statistic,
                                       rtol=1e-9, err_msg=lp.role)


def test_workbook_lists_skipped_pairs_even_when_no_pair_ran(tmp_path):
    from hdeeg_snpm.report.excel import write_workbook
    from hdeeg_snpm.views import EffectView
    z = np.zeros(3)
    view = EffectView("main", "group (3 levels)", "F", z, z, z, z, "mass", "p < 0.05", [], 10,
                      np.empty(0, int), skipped=[("a vs b", "cannot be estimated")])
    write_workbook(tmp_path / "w.xlsx", [view], ["E1", "E2", "E3"], 0.05)
    contents = pd.read_excel(tmp_path / "w.xlsx", sheet_name="contents")
    assert contents.Status.tolist() == ["run", "not run: cannot be estimated"]
