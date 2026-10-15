import numpy as np
import pytest

from hdeeg_snpm.clusters import ChannelGraph, find_clusters
from hdeeg_snpm.montage import get_montage, resolve_key, setup_channels


@pytest.mark.parametrize("key,n", [("egi", 178), ("compu", 249)])
def test_montage_assets(key, n):
    m = get_montage(key)
    assert len(m.labels) == n == len(set(m.labels))
    assert m.neighbors.shape[0] == n and m.theta.shape == m.radius.shape == (n,)
    g = ChannelGraph(m.neighbors)
    assert (g.dense == g.dense.T).all() and not g.dense.diagonal().any()
    assert g.dense.any(axis=1).all(), "every channel has at least one neighbour"


def test_aliases():
    assert resolve_key("178 channels") == resolve_key("EGI256") == "egi"
    assert resolve_key("Neuvo") == "compu"
    with pytest.raises(ValueError):
        resolve_key("biosemi")


def test_setup_keeps_scalp_channels_in_data_order():
    cols = ["Subject"] + [f"E{i}" for i in range(256, 0, -1)] + ["Cz"]
    data = np.arange(3 * len(cols), dtype=float).reshape(3, len(cols))
    s = setup_channels("egi", [data], cols)
    assert len(s.labels) == 178 and s.labels[-1] == "Cz"
    assert list(s.labels) == [cols[j] for j in s.data_columns]
    assert (s.data[0] == data[:, s.data_columns]).all()
    # the reordered graph is the montage graph with rows and columns permuted
    full = ChannelGraph(get_montage("egi").neighbors).dense
    assert (s.graph.dense == full[np.ix_(s.montage_indices, s.montage_indices)]).all()


def test_setup_subset_drops_missing_neighbours():
    m = get_montage("compu")
    keep = list(m.labels[:40])
    s = setup_channels("compu", [np.zeros((2, 40))], keep)
    full = ChannelGraph(m.neighbors).dense
    assert (s.graph.dense == full[:40, :40]).all()
    with pytest.raises(ValueError, match="No data columns"):
        setup_channels("compu", [np.zeros((2, 2))], ["E1", "E2"])


def test_find_clusters(egi_graph):
    values = np.zeros(egi_graph.n)
    a = [0] + egi_graph.neighbor_lists[0]
    far = next(i for i in range(egi_graph.n)
               if i not in a and not set(egi_graph.neighbor_lists[i]) & set(a))
    values[a] = 3.0
    values[far] = -4.0
    values[5 if 5 not in a + [far] else 6] = np.nan
    pos = find_clusters(values, 2.0, egi_graph)
    assert len(pos) == 1 and sorted(pos[0]) == sorted(a)
    neg = find_clusters(values, -2.0, egi_graph)
    assert len(neg) == 1 and neg[0].tolist() == [far]
    assert find_clusters(values, 5.0, egi_graph) == []
