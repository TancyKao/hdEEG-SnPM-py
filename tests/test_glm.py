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
    # site is x for a, y for b and alternates within c: it coincides with group on the a and b
    # rows only, so the omnibus effect is estimable and only the pair a vs b is not. (Site equal
    # to "a versus the rest" aliases group_c in the omnibus, which is now refused: see
    # test_effect_aliased_with_the_nuisance_is_refused.)
    c_rows = np.flatnonzero(meta.group == "c")
    site = np.where(meta.group == "a", "x", "y").astype(object)
    site[c_rows[::2]] = "x"
    meta["site"] = site
    with pytest.warns(UserWarning, match="a vs b.*cannot be estimated"):
        plan = glm.plan_preset("ancova", Y, meta, group_col="group", covariate_cols=["site"])
    roles = [lp.role for lp in plan]
    assert "main/posthoc:a vs b" not in roles
    assert "main/posthoc:a vs c" in roles and "main/posthoc:b vs c" in roles
    assert [pair for pair, _ in plan[0].skipped] == ["a vs b"]


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


# ------------------------------------------- MATLAB test_glm_snpm T19-T22 (2026-10-26): estimability

_ALIAS_HEAD = "The effect tested ({}) cannot be separated from the nuisance columns of the design: "
_ALIAS_TAIL = (". Its contrast is not estimable, so the statistic and every p-value would be "
               "arbitrary. Remove the covariate that repeats the effect, or test a different "
               "effect.")


def _alias_meta(seed=7):
    rng = np.random.default_rng(seed)
    n = 30
    logahi, age = rng.standard_normal(n), 40 + 10 * rng.standard_normal(n)
    g = np.tile(["a", "b", "c"], n // 3)
    return pd.DataFrame({"logAHI": logahi, "age": age, "ahi2": 2 * logahi + 5,
                         "age_months": age * 12, "group": g,
                         "site": np.where(g == "a", "x", "y")})       # site_y = D_b + D_c


def _alias_csv(tmp_path, name, meta, rng):
    from hdeeg_snpm.montage import get_montage
    labels = list(get_montage("egi").labels)
    frame = pd.concat([meta, pd.DataFrame(rng.standard_normal((len(meta), len(labels))),
                                          columns=labels)], axis=1)
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def _glm_params(preset, path, meta_cols, perms=20, **kw):
    return dict({"comparison": preset, "data_file": path, "channels": "egi",
                 "meta_cols": list(meta_cols), "permutations": perms, "seed": 1}, **kw)


def test_effect_aliased_with_the_nuisance_is_refused(tmp_path):
    """MATLAB T19: an effect in the span of the nuisance columns has no estimable contrast; pinv
    would give a finite, arbitrary statistic. Refused with the aliased columns named."""
    from hdeeg_snpm import pipeline
    meta = _alias_meta()
    path = _alias_csv(tmp_path, "glm_alias", meta, np.random.default_rng(1))
    cases = [("regression", {"predictor_col": "logAHI", "covariate_cols": ["logAHI", "age"]},
              "slope of logAHI", "predictor logAHI is a linear combination of covariate logAHI"),
             ("regression", {"predictor_col": "logAHI", "covariate_cols": ["ahi2", "age"]},
              "slope of logAHI", "predictor logAHI is a linear combination of covariate ahi2"),
             ("ancova", {"group_col": "group", "covariate_cols": ["age", "site"]},
              "group (3 levels), covariate-adjusted",
              "group_c is a linear combination of (Intercept), covariate site_y, group_b")]
    for preset, kw, label, item in cases:
        with pytest.raises(glm.GLMInputError) as e:
            pipeline.run_glm(_glm_params(preset, path, meta.columns, **kw))
        assert e.value.code == "effectAliased"
        assert str(e.value) == _ALIAS_HEAD.format(label) + item + _ALIAS_TAIL

    # rmanova: condition r only in two subjects with a single row each, so its dummy is the sum
    # of their subject dummies
    mr = pd.DataFrame({"subject": np.r_[np.repeat(np.arange(1, 9), 2), 9, 10],
                       "condition": list(np.tile(["p", "q"], 8)) + ["r", "r"]})
    path = _alias_csv(tmp_path, "glm_alias_rm", mr, np.random.default_rng(2))
    with pytest.raises(glm.GLMInputError) as e:
        pipeline.run_glm(_glm_params("rmanova", path, mr.columns, subject_col="subject",
                                     condition_col="condition"))
    assert e.value.code == "effectAliased"
    assert str(e.value) == (_ALIAS_HEAD.format("condition (3 levels), within-subject")
                            + "condition_r is a linear combination of subject_9, subject_10"
                            + _ALIAS_TAIL)


def test_nuisance_only_dependency_warns_once_and_changes_nothing(tmp_path):
    """MATLAB T20: covariates {age, age_months} span the space of {age}: the effect is
    estimable, the run warns once and its maps equal the {age} run's."""
    import warnings

    from hdeeg_snpm import pipeline
    meta = _alias_meta()
    path = _alias_csv(tmp_path, "glm_nuis_rank", meta, np.random.default_rng(20))
    base = _glm_params("regression", path, meta.columns, perms=50, predictor_col="logAHI")
    with warnings.catch_warnings(record=True) as w1:
        warnings.simplefilter("always")
        r1 = pipeline.run_glm(dict(base, covariate_cols=["age", "age_months"]))
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        r2 = pipeline.run_glm(dict(base, covariate_cols=["age"]))
    hits = [x for x in w1 if issubclass(x.category, glm.GLMNuisanceRankDeficient)]
    assert len(hits) == 1
    assert str(hits[0].message) == (
        "The nuisance columns of the design are linearly dependent: covariate age_months is a "
        "linear combination of covariate age. The effect is still estimable and the analysis "
        "runs; the adjustment is for the space these columns span, which the redundant column "
        "does not enlarge. Check the covariate list.")
    assert not [x for x in w2 if issubclass(x.category, glm.GLMNuisanceRankDeficient)]
    a, b = r1.loop("main").result, r2.loop("main").result
    assert np.array_equal(np.isnan(a.real_T), np.isnan(b.real_T))
    assert np.nanmax(np.abs(a.real_T - b.real_T)) <= 1e-12 * max(1, np.nanmax(np.abs(b.real_T)))
    assert np.array_equal(a.p_correctedTFCE, b.p_correctedTFCE, equal_nan=True)

    # a post-hoc pair with the redundant covariate runs, with the t of the pair without it
    Y = np.random.default_rng(3).standard_normal((len(meta), 5))
    plans = [glm.plan_preset("ancova", Y, meta, group_col="group", covariate_cols=cv)
             for cv in (["age", "age_months"], ["age"])]
    for plan in plans:
        assert not plan[0].skipped
    p1, p2 = (next(lp for lp in plan if lp.role == "main/posthoc:a vs b") for plan in plans)
    assert np.max(np.abs(glm.glm_stat(p1.Y, p1.design.X, p1.C)[0]
                         - glm.glm_stat(p2.Y, p2.design.X, p2.C)[0])) < 1e-10   # pinv on a
    # rank-deficient X agrees to about 2e-12 here (MATLAB's own check uses 1e-12)


def test_clean_regression_is_estimable_and_recovers_a_planted_effect():
    """MATLAB T21: the full-rank design is not flagged, has MATLAB's column names, warns
    nothing, recovers a planted effect and finds nothing under the null."""
    import warnings

    from hdeeg_snpm.clusters import ChannelGraph
    meta = _alias_meta()
    n, n_ch = len(meta), 20
    nb = np.full((n_ch, 2), -1)
    nb[1:, 0] = np.arange(n_ch - 1)
    nb[:-1, 1] = np.arange(1, n_ch)
    graph = ChannelGraph(nb)                                     # a chain of 20 channels
    d = glm.build_design("regression", meta, predictor_col="logAHI", covariate_cols=["age"])
    info = glm.estimability(d)
    assert info.estimable and not info.nuisance_deficient
    assert d.col_names == ["(Intercept)", "predictor logAHI", "covariate age"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        glm.estimability(d, enforce=True)
    rng = np.random.default_rng(21)
    Y = rng.standard_normal((n, n_ch))
    Y[:, :5] += meta["logAHI"].to_numpy()[:, None]
    hit = glm.run_effect(Y, d, graph, n_perm=500, rng=rng).p_correctedTFCE < 0.05
    null = glm.run_effect(np.random.default_rng(22).standard_normal((n, n_ch)), d, graph,
                          n_perm=500, rng=np.random.default_rng(23)).p_correctedTFCE < 0.05
    assert hit[:5].all() and not hit[5:].any()
    assert not null.any()


def test_ancova_without_a_covariate_is_refused_before_the_file_is_read(tmp_path):
    """MATLAB T22: ANCOVA with no covariate is a one-way ANOVA labelled covariate-adjusted."""
    from hdeeg_snpm import pipeline
    want = "ANCOVA needs at least one covariate; with none, use anova1."
    absent = str(tmp_path / "not_there.csv")             # never read
    for extra in ({}, {"covariate_cols": []}):
        with pytest.raises(glm.GLMInputError) as e:
            pipeline.run_glm(_glm_params("ancova", absent, ["group"], group_col="group", **extra))
        assert e.value.code == "ancovaNoCovariate" and str(e.value) == want


def test_ancova_blank_or_dropped_covariates_are_refused(tmp_path, egi_graph):
    """MATLAB snpm_glm_require_covariate: '' and [''] name no covariate (refused before the
    file is read); covariates that are all constant on the analysed rows are dropped by the
    design, which leaves no covariate (refused after reading). The check also guards the
    engine entry points used without the pipeline."""
    from hdeeg_snpm import pipeline
    want = "ANCOVA needs at least one covariate; with none, use anova1."
    absent = str(tmp_path / "not_there.csv")
    for blank in ("", [""], ["", "  "]):
        with pytest.raises(glm.GLMInputError) as e:
            pipeline.run_glm(_glm_params("ancova", absent, ["group"], group_col="group",
                                         covariate_cols=blank))
        assert e.value.code == "ancovaNoCovariate" and str(e.value) == want

    meta = _alias_meta().assign(const=5.0, ward="w1")      # numeric constant, one-level text
    path = _alias_csv(tmp_path, "glm_ancova_const", meta, np.random.default_rng(4))
    for cv in (["const"], ["ward"], ["const", "ward"]):
        with pytest.raises(glm.GLMInputError) as e:
            pipeline.run_glm(_glm_params("ancova", path, meta.columns, group_col="group",
                                         covariate_cols=cv))
        assert e.value.code == "ancovaNoCovariate" and str(e.value) == want
    # one varying covariate among constant ones runs
    res = pipeline.run_glm(_glm_params("ancova", path, meta.columns, group_col="group",
                                       covariate_cols=["const", "age"]))
    assert res.loop("main").design.col_names[-1] == "covariate age"

    # engine entry points without the pipeline: no covariate given, or none kept
    Y = np.random.default_rng(5).standard_normal((len(meta), egi_graph.n))
    for cv in (None, ["const"]):
        with pytest.raises(glm.GLMInputError) as e:
            glm.plan_preset("ancova", Y, meta, group_col="group", covariate_cols=cv)
        assert e.value.code == "ancovaNoCovariate"
    with pytest.raises(glm.GLMInputError) as e:          # and the estimability refusal
        glm.plan_preset("regression", Y, meta, predictor_col="logAHI",
                        covariate_cols=["logAHI"])
    assert e.value.code == "effectAliased"


def test_boolean_factor_levels_are_named_as_matlab_writes_them():
    """snpm_glm_design.m tostr writes a logical level as false/true (lowercase), so an aliased
    boolean group level reads g_false in the column names and in the refusal."""
    g = np.tile([True, False], 10)                     # True first: reference True, dummy g_false
    meta = pd.DataFrame({"g": g, "same": (~g).astype(float), "age": np.arange(20.0)})
    d = glm.build_design("ancova", meta, group_col="g", covariate_cols=["age"])
    assert d.col_names == ["(Intercept)", "g_false", "covariate age"]
    with pytest.raises(glm.GLMInputError) as e:
        glm.estimability(glm.build_design("ancova", meta, group_col="g",
                                          covariate_cols=["same"]), enforce=True)
    assert e.value.code == "effectAliased"
    assert "g_false is a linear combination of (Intercept), covariate same" in str(e.value)


def test_a_text_predictor_is_refused(tmp_path):
    """MATLAB core_snpm_glm:predictorNotNumeric: a regression slope needs a numeric predictor.
    A text column is refused before the design is built, even when its values look like numbers
    (read_table already reads a numeric column, blanks included, as numbers). A true/false
    column is refused as MATLAB's readtable reads it as text; the value quoted is the first not
    missing or blank, a boolean in lowercase as MATLAB prints a logical, text as read. Numeric
    columns pass."""
    from pathlib import Path

    from hdeeg_snpm import pipeline
    example = Path(__file__).resolve().parent.parent / "examples" / "data" / "glm_regression.csv"
    meta_cols = ["Subject", "group", "logAHI", "age", "sex"]
    with pytest.raises(glm.GLMInputError) as e:
        pipeline.run_glm(_glm_params("regression", str(example), meta_cols,
                                     predictor_col="group", covariate_cols=["age"]))
    assert e.value.code == "predictorNotNumeric" and str(e.value) == (
        "The predictor column 'group' is not numeric (it holds text such as 'OSA'); regression "
        "needs a numeric predictor. To compare groups, use anova1 or ancova.")

    n = 12
    meta = pd.DataFrame({"x": np.arange(n, dtype=float), "flag": np.tile([True, False], n // 2),
                         "txt": [None, "  "] + [str(v) for v in range(2, n)]})
    meta["flag_text"] = np.where(meta["flag"], "True", "False").astype(object)
    for col, first in (("txt", "2"), ("flag", "true"), ("flag_text", "True")):
        with pytest.raises(glm.GLMInputError) as e:
            glm.build_design("regression", meta, predictor_col=col)
        assert e.value.code == "predictorNotNumeric" and str(e.value) == (
            f"The predictor column '{col}' is not numeric (it holds text such as '{first}'); "
            "regression needs a numeric predictor. To compare groups, use anova1 or ancova.")
    assert glm.build_design("regression", meta, predictor_col="x").col_names == \
        ["(Intercept)", "predictor x"]

    csv = tmp_path / "blank.csv"                          # a blank cell: still numeric
    csv.write_text("Subject,x\ns1,1.5\ns2,\ns3,2\n")
    from hdeeg_snpm.io.tables import read_table
    assert pd.api.types.is_float_dtype(read_table(csv)["x"])
    csv.write_text("Subject,flag\ns1,True\ns2,False\n")    # pandas reads it as bool: refused
    with pytest.raises(glm.GLMInputError) as e:
        glm.build_design("regression", read_table(csv), predictor_col="flag")
    assert e.value.code == "predictorNotNumeric" and "text such as 'true'" in str(e.value)


def test_covariate_names_are_stripped_and_trimmed_as_matlab_does():
    """snpm_glm_covariate_list.m: blank names dropped, kept names trimmed."""
    assert glm.covariate_list([" age ", "", "  "]) == ["age"]
    assert glm.covariate_list(["age", "sex"]) == ["age", "sex"]
    assert glm.covariate_list(None) == [] and glm.covariate_list("") == []
