"""t-test and correlation engines: statistics against scipy, planted-cluster recovery, controls."""

import numpy as np
import pytest
from scipy import stats

from hdeeg_snpm import legacy


def planted(egi_graph, n=20, effect=1.5, seed=0, missing=0.0):
    rng = np.random.default_rng(seed)
    cluster = sorted({23, *egi_graph.neighbor_lists[23]})
    x = rng.standard_normal((n, egi_graph.n))
    y = rng.standard_normal((n, egi_graph.n))
    x[:, cluster] += effect
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    return x, y, cluster


def test_statistics_match_scipy(egi_graph):
    x, y, _ = planted(egi_graph, missing=0.1)
    t, df = legacy.paired_t(x - y)
    ref = stats.ttest_rel(x, y, nan_policy="omit")
    np.testing.assert_allclose(t, ref.statistic, rtol=1e-10)
    np.testing.assert_allclose(legacy._t_pvalue(t, df, True), ref.pvalue, rtol=1e-9)

    t, df = legacy.unpaired_t(x, y[:13])
    ref = stats.ttest_ind(x, y[:13], nan_policy="omit")
    np.testing.assert_allclose(t, ref.statistic, rtol=1e-10)
    np.testing.assert_allclose(legacy._t_pvalue(t, df, True), ref.pvalue, rtol=1e-9)

    xc, yc, _ = planted(egi_graph, seed=3)
    r = legacy.corr_columns(xc, yc)
    np.testing.assert_allclose(r, [stats.pearsonr(xc[:, j], yc[:, j])[0] for j in range(xc.shape[1])],
                               rtol=1e-10)
    rs = legacy.corr_columns(legacy.tiedrank(xc), legacy.tiedrank(yc))
    np.testing.assert_allclose(rs, [stats.spearmanr(xc[:, j], yc[:, j])[0] for j in range(xc.shape[1])],
                               rtol=1e-10)


def test_corr_columns_complete_column_rule():
    rng = np.random.default_rng(0)
    x, y = rng.standard_normal((10, 4)), rng.standard_normal((10, 4))
    x[2, 0] = np.nan
    y[5, 1] = -np.inf
    x[:, 2] = 1.0
    r = legacy.corr_columns(x, y)
    assert np.isnan(r[:3]).all() and np.isfinite(r[3])


@pytest.mark.parametrize("comparison", ["pairedT", "unpairedT", "onesampleT"])
def test_planted_cluster_recovered(egi_graph, comparison):
    x, y, cluster = planted(egi_graph)
    if comparison == "onesampleT":
        y = np.zeros_like(x)
    rng = np.random.default_rng(1)
    res = legacy.single_threshold_tfce(x, y, egi_graph, comparison, n_perm=300, rng=rng)
    sig = np.flatnonzero(res.p_correctedTFCE < 0.05)
    assert set(cluster) <= set(sig) and len(sig) <= len(cluster) + 4
    thr = stats.t.ppf(0.975, x.shape[0] - 1)
    cl = legacy.cluster_analysis(x, y, thr, egi_graph, comparison, n_perm=300, rng=rng)
    best = min(cl.clusters, key=lambda c: c.p)
    assert best.p < 0.05 and set(cluster) <= set(best.channels) and best.threshold > 0


def test_observed_labelling_is_in_the_null(egi_graph):
    x, y, _ = planted(egi_graph, effect=0.0)
    for comparison in ("pairedT", "unpairedT", "correlationP", "correlationS"):
        res = legacy.single_threshold_tfce(x, y, egi_graph, comparison, n_perm=50,
                                           rng=np.random.default_rng(2))
        assert res.tMax.size == 50 and (np.diff(res.tMax) <= 0).all()
        # row 0 of the permutation matrix is the observed labelling, so its maximum is in the null
        assert np.nanmax(np.abs(res.real_T)) in res.tMax
        assert res.p_corrected.min() >= 2 / 51


def test_null_design_rarely_rejects(egi_graph):
    hits = 0
    for seed in range(40):
        x, y, _ = planted(egi_graph, n=12, effect=0.0, seed=100 + seed)
        res = legacy.single_threshold_tfce(x, y, egi_graph, "pairedT", n_perm=100,
                                           rng=np.random.default_rng(seed))
        hits += res.p_correctedTFCE.min() < 0.05
    assert hits <= 6, f"{hits}/40 null datasets rejected at 0.05"


def test_left_tail_is_right_tail_on_swapped_inputs(egi_graph):
    x, y, cluster = planted(egi_graph)
    perms = np.random.default_rng(0).random((60, x.shape[0])) > 0.5
    perms[0] = True
    left = legacy.single_threshold_tfce(y, x, egi_graph, "pairedT", tail="left", perms=perms)
    right = legacy.single_threshold_tfce(x, y, egi_graph, "pairedT", tail="right", perms=perms)
    np.testing.assert_array_equal(left.real_T, right.real_T)
    np.testing.assert_array_equal(left.p_correctedTFCE, right.p_correctedTFCE)
    assert (right.p_correctedTFCE[cluster] < 0.05).all()


def test_correlation_recovers_planted_association(egi_graph):
    rng = np.random.default_rng(5)
    n = 30
    cluster = sorted({23, *egi_graph.neighbor_lists[23]})
    score = rng.standard_normal(n)
    x = rng.standard_normal((n, egi_graph.n))
    x[:, cluster] += 1.5 * score[:, None]
    y = np.repeat(score[:, None], egi_graph.n, axis=1)
    for comparison in ("correlationP", "correlationS"):
        res = legacy.single_threshold_tfce(x, y, egi_graph, comparison, n_perm=300, rng=rng)
        assert set(cluster) <= set(np.flatnonzero(res.p_correctedTFCE < 0.05))
        cl = legacy.cluster_analysis(x, y, 0.40, egi_graph, comparison, n_perm=300, rng=rng)
        assert min(c.p for c in cl.clusters) < 0.05


def test_supplied_permutations_are_used_and_reproducible(egi_graph):
    x, y, _ = planted(egi_graph)
    a = legacy.single_threshold_tfce(x, y[:15], egi_graph, "unpairedT", n_perm=40,
                                     rng=np.random.default_rng(9))
    b = legacy.single_threshold_tfce(x, y[:15], egi_graph, "unpairedT", perms=a.perms)
    np.testing.assert_array_equal(a.tMaxTFCE, b.tMaxTFCE)
    np.testing.assert_array_equal(a.p_correctedTFCE, b.p_correctedTFCE)


def test_bad_arguments(egi_graph):
    x, y, _ = planted(egi_graph, n=6)
    with pytest.raises(ValueError, match="unsupported comparison"):
        legacy.single_threshold_tfce(x, y, egi_graph, "anova1")
    with pytest.raises(ValueError, match="tail='both' only"):
        legacy.single_threshold_tfce(x, y, egi_graph, "correlationP", tail="left")


def test_tie_counting_rule():
    """Counting null >= observed treats a relative 1e-12 as a tie (round 4, question 2)."""
    null = np.array([1.0 + 4e-14, 1.0 - 4e-14, 1.0 - 1e-9, 0.0, np.nan])
    assert legacy._at_least(null, 1.0).tolist() == [True, True, False, False, False]
    assert legacy._at_least(np.array([0.0]), 0.0).tolist() == [True]
    # a NaN null maximum is dropped from the count but kept in N
    p = legacy._fwe_p(np.array([1.0, np.nan]), null, two_sided=False)
    assert p[0] == 3 / 6 and np.isnan(p[1])


# ------------------------------------------------- ruling D: observed labelling counted once

def _one_channel():
    from hdeeg_snpm.clusters import ChannelGraph
    return ChannelGraph(np.full((1, 1), -1))


def test_exact_paired_p_counts_observed_once():
    """4 subjects, distinct positive differences, right tail, all 16 sign flips: p = 1/16."""
    x = np.array([[1.0], [2.0], [3.5], [5.0]])
    y = np.zeros_like(x)
    g = _one_channel()
    res = legacy.single_threshold_tfce(x, y, g, "pairedT", tail="right")
    assert res.n_perm == 16
    assert res.p_corrected[0] == 1 / 16 and res.p_correctedTFCE[0] == 1 / 16
    thr = (res.tMax[0] + res.tMax[1]) / 2           # only the observed labelling forms a cluster
    cl = legacy.cluster_analysis(x, y, thr, g, "pairedT", tail="right")
    assert [c.p for c in cl.clusters] == [1 / 16]


def test_exact_unpaired_two_sided_mirror_ties():
    """3 vs 3 perfectly separated, two-sided, all 20 splits: the mirror ties, p = 2/20."""
    x = np.array([[10.0], [11.0], [12.5]])
    y = np.array([[0.0], [1.5], [2.0]])
    g = _one_channel()
    res = legacy.single_threshold_tfce(x, y, g, "unpairedT", tail="both")
    assert res.n_perm == 20
    assert res.p_corrected[0] == 2 / 20 and res.p_correctedTFCE[0] == 2 / 20
    thr = (res.tMax[1] + res.tMax[2]) / 2           # only the observed split and its mirror
    cl = legacy.cluster_analysis(x, y, thr, g, "unpairedT", tail="both")
    assert [c.p for c in cl.clusters if c.channels.size] == [2 / 20]


@pytest.mark.parametrize("comparison", ["pairedT", "unpairedT", "correlationP"])
def test_monte_carlo_p_is_phipson_smyth_over_random_rows(egi_graph, comparison):
    """Random draws: p = (b' + 1) / N, b' counting rows 2..N (row 1 is the observed labelling)."""
    x, y, _ = planted(egi_graph, n=12, effect=0.8, seed=5)
    x, y = x[:, :20], y[:, :20]
    from hdeeg_snpm.clusters import ChannelGraph
    sub = ChannelGraph(np.where(egi_graph.dense[:20, :20], np.arange(20)[None, :], -1))
    n_perm = 150
    res = legacy.single_threshold_tfce(x, y, sub, comparison, n_perm=n_perm,
                                       rng=np.random.default_rng(1))
    design = legacy._design(x, y, comparison, "both")
    rows = res.perms
    assert n_perm < design.default_n_perm()
    null = np.array([np.nanmax(np.abs(design.stat(r))) for r in rows])
    obs = np.abs(res.real_T)
    assert np.isclose(null[0], np.nanmax(obs), rtol=1e-12)        # row 1 is the observed one
    b_random = legacy._at_least(null[1:][None, :], obs[:, None]).sum(axis=1)
    np.testing.assert_array_equal(res.p_corrected, (b_random + 1) / n_perm)


def test_exact_null_calibration():
    """Paired, 6 subjects, exact (64 flips), H0: rejection rate at k/64 is k/64.

    The old (b + 1) / (N + 1) gave 3/64 at alpha = 4/64.
    """
    from hdeeg_snpm import perm
    rng = np.random.default_rng(20261020)
    n_sim, n_subj = 20000, 6
    signs = np.where(perm.signflip_patterns(n_subj, 64, rng), 1.0, -1.0)   # (64, 6)
    d = rng.standard_normal((n_sim, n_subj))
    flipped = d[:, None, :] * signs[None, :, :]                           # (sim, 64, 6)
    t = flipped.mean(-1) / (flipped.std(-1, ddof=1) / np.sqrt(n_subj))
    obs = t[:, -1]                                                         # all-True row
    p = np.array([legacy._fwe_p(obs[i:i + 1], np.sort(t[i])[::-1], False,
                                identity_in_null=True)[0] for i in range(n_sim)])
    se = np.sqrt((4 / 64) * (60 / 64) / n_sim)
    assert abs(np.mean(p <= 4 / 64) - 4 / 64) < 0.004
    for k in range(1, 9):
        a = k / 64
        assert np.mean(p <= a + 1e-12) <= a + 4 * se
        assert abs(np.mean(p <= a + 1e-12) - a) < 4 * np.sqrt(a * (1 - a) / n_sim)


def _identity_free_perms(comparison, n):
    """A supplied labelling set that lacks the observed labelling."""
    if comparison == "pairedT":
        return np.array([[False, True, True, True], [True, False, True, True],
                         [True, True, False, True]])
    if comparison == "unpairedT":                       # 3 vs 3: group 1 never {0, 1, 2}
        return np.array([[3, 4, 5, 0, 1, 2], [0, 1, 3, 2, 4, 5], [1, 2, 4, 0, 3, 5]])
    rng = np.random.default_rng(0)
    rows = [rng.permutation(n) for _ in range(5)]
    return np.array([r for r in rows if not np.array_equal(r, np.arange(n))])


@pytest.mark.parametrize("comparison,n1,n2", [("pairedT", 4, 4), ("unpairedT", 3, 3),
                                              ("correlationP", 6, 6)])
def test_supplied_perms_without_observed_labelling_are_refused(comparison, n1, n2):
    """b / N needs the observed labelling among the rows; a matrix without it is refused."""
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((n1, 1)) + 1.0, rng.standard_normal((n2, 1))
    g = _one_channel()
    perms = _identity_free_perms(comparison, n1)
    tail = "right" if comparison == "pairedT" else "both"
    with pytest.raises(ValueError, match="must contain the observed labelling"):
        legacy.single_threshold_tfce(x, y, g, comparison, tail=tail, perms=perms)
    with pytest.raises(ValueError, match="must contain the observed labelling"):
        legacy.cluster_analysis(x, y, 1.0, g, comparison, tail=tail, perms=perms)


@pytest.mark.parametrize("comparison,n1,n2", [("pairedT", 4, 4), ("unpairedT", 3, 3),
                                              ("correlationP", 6, 6)])
def test_supplied_perms_with_observed_labelling_are_accepted(comparison, n1, n2):
    """MATLAB's exported matrices hold the identity (sign flips: the last enumerated row;
    relabelling: row 1, possibly with group 1 in another order; orderings: row 1)."""
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal((n1, 1)) + 1.0, rng.standard_normal((n2, 1))
    g = _one_channel()
    perms = _identity_free_perms(comparison, n1)
    identity = {"pairedT": np.ones(4, bool), "unpairedT": np.array([2, 0, 1, 3, 4, 5]),
                "correlationP": np.arange(6)}[comparison]
    perms = np.vstack([perms, identity[None, :].astype(perms.dtype)])
    tail = "right" if comparison == "pairedT" else "both"
    res = legacy.single_threshold_tfce(x, y, g, comparison, tail=tail, perms=perms)
    assert np.all(res.p_corrected >= 1 / perms.shape[0])
    legacy.cluster_analysis(x, y, 1.0, g, comparison, tail=tail, perms=perms)


def test_report_states_the_p_convention_of_each_tier():
    from hdeeg_snpm.report.html import _p_text
    for comparison in legacy.COMPARISONS:
        assert _p_text(comparison).startswith("p-values are b / N, where N counts the observed")
    for comparison in ("anova1", "ancova", "regression", "rmanova", "mixed2way", "mixedmodel",
                       "circ_phase_group", "circ_phase_group_u2", "circ_corrAngLinear"):
        assert _p_text(comparison).startswith("p-values are (b + 1) / (N + 1)")
