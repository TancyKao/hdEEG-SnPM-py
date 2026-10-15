"""Independent check against MNE-Python for the designs it can express.

MNE draws its own permutations and scores clusters by mass, so only the observed statistic map
and the membership of the observed clusters are compared.
"""

import numpy as np
import pytest
from scipy import stats

from hdeeg_snpm import legacy

mne_stats = pytest.importorskip("mne.stats")


def _data(egi_graph, seed=0, n=18):
    rng = np.random.default_rng(seed)
    cluster = sorted({23, *egi_graph.neighbor_lists[23]})
    x = rng.standard_normal((n, egi_graph.n))
    y = rng.standard_normal((n, egi_graph.n))
    x[:, cluster] += 1.5
    return x, y


def _sets(clusters):
    return {frozenset(int(i) for i in np.atleast_1d(c)) for c in clusters if np.size(c)}


def test_paired_matches_mne(egi_graph):
    x, y = _data(egi_graph)
    thr = stats.t.ppf(0.975, x.shape[0] - 1)
    t_obs, clusters, _, _ = mne_stats.permutation_cluster_1samp_test(
        x - y, threshold=thr, adjacency=egi_graph.csr, n_permutations=10, out_type="indices",
        seed=0, verbose=False)
    ours = legacy.cluster_analysis(x, y, thr, egi_graph, "pairedT", n_perm=10,
                                   rng=np.random.default_rng(0))
    np.testing.assert_allclose(ours.real_stat, t_obs, rtol=1e-10)
    assert _sets(c.channels for c in ours.clusters) == _sets(c[0] for c in clusters)


def test_unpaired_matches_mne(egi_graph):
    x, y = _data(egi_graph, seed=1)
    y = y[:12]
    thr = stats.t.ppf(0.975, x.shape[0] + y.shape[0] - 2)
    t_obs, clusters, _, _ = mne_stats.permutation_cluster_test(
        [x, y], threshold=thr, tail=0, adjacency=egi_graph.csr, n_permutations=10,
        stat_fun=lambda a, b: stats.ttest_ind(a, b).statistic, out_type="indices", seed=0,
        verbose=False)
    ours = legacy.cluster_analysis(x, y, thr, egi_graph, "unpairedT", n_perm=10,
                                   rng=np.random.default_rng(0))
    np.testing.assert_allclose(ours.real_stat, t_obs, rtol=1e-10)
    assert _sets(c.channels for c in ours.clusters) == _sets(c[0] for c in clusters)
