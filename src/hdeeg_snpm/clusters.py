"""Channel-neighbour graph and suprathreshold clusters.

Ports ``make_neighbors_sparse.m`` and ``snpm_find_clusters_graphalgs.m``.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


class ChannelGraph:
    """Undirected channel-adjacency graph built from a neighbour matrix.

    Parameters
    ----------
    neighbors : (n_channels, max_neighbours) int array
        Row ``i`` lists the 0-based indices of channel ``i``'s neighbours, padded with -1.
        The matrix need not be symmetric; an edge listed in either direction is kept in both,
        as in ``make_neighbors_sparse.m``.
    """

    def __init__(self, neighbors: np.ndarray):
        neighbors = np.asarray(neighbors)
        n = neighbors.shape[0]
        rows, cols = np.nonzero(neighbors >= 0)
        targets = neighbors[rows, cols]
        dense = np.zeros((n, n), dtype=bool)
        dense[rows, targets] = True
        dense[targets, rows] = True
        self.n = n
        self.dense = dense
        self.csr = sparse.csr_matrix(dense)
        self.neighbor_lists = [np.flatnonzero(dense[i]).tolist() for i in range(n)]


def find_clusters(values: np.ndarray, threshold: float, graph: ChannelGraph) -> list[np.ndarray]:
    """Connected sets of channels beyond ``threshold``.

    A positive threshold selects ``values > threshold``; a zero or negative one selects
    ``values < threshold``. NaN channels are never selected. Each cluster is a sorted array of
    0-based channel indices, and clusters are returned in order of their lowest channel index.
    (MATLAB returns them in ``dmperm`` block order, which is not reproduced; callers that report
    cluster numbers order them explicitly.)
    """
    values = np.asarray(values, dtype=float)
    mask = values > threshold if threshold > 0 else values < threshold
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    sub = graph.csr[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False)
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], np.arange(n_comp + 1))
    clusters = [idx[order[bounds[k]:bounds[k + 1]]] for k in range(n_comp)]
    clusters.sort(key=lambda c: c[0])
    return clusters
