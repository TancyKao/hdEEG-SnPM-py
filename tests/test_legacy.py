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
