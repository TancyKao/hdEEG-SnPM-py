"""Permutation index generators.

Ports ``snpm_signflip_patterns.m``, ``snpm_relabel_assignments.m``, ``snpm_row_permutations.m``,
``enumerate_combinations.m`` and the index part of ``snpm_glm_permute.m``.

Every generator returns the whole index matrix (one row per permutation, 0-based), so an engine
can be handed a matrix from elsewhere instead, in particular the one MATLAB used. The exact-
enumeration branches list rows in MATLAB's order. The random branches use a
``numpy.random.Generator`` and therefore do not reproduce MATLAB's draws for the same seed.
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import comb, factorial

import numpy as np


def signflip_patterns(n_subj: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Sign-flip labellings for paired and one-sample tests: (n_perm, n_subj) bool.

    ``True`` keeps a subject's difference, ``False`` flips its sign. If all ``2**n_subj``
    patterns fit in ``n_perm`` they are enumerated (row ``k`` is the binary digits of ``k``,
    lowest bit first, so the observed all-``True`` labelling is the last row). Otherwise each
    entry is an independent fair coin and row 0 is forced to the observed labelling.
    """
    if 2 ** n_subj <= n_perm:
        k = np.arange(2 ** n_subj)[:, None]
        return ((k >> np.arange(n_subj)) & 1).astype(bool)
    patterns = rng.random((n_perm, n_subj)) > 0.5
    patterns[0] = True
    return patterns


def enumerate_combinations(n_subj: int, n_grp1: int) -> np.ndarray:
    """Every split of ``n_subj`` subjects into a first group of ``n_grp1`` and the rest.

    Row = [group-1 indices ascending, group-2 indices ascending]; rows are in lexicographic
    order of the group-1 indices, so row 0 is the observed split.
    """
    everyone = range(n_subj)
    rows = [g1 + tuple(i for i in everyone if i not in g1) for g1 in combinations(everyone, n_grp1)]
    return np.array(rows, dtype=int)


def _random_orders(n: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    orders = np.empty((n_perm, n), dtype=int)
    orders[0] = np.arange(n)
    for r in range(1, n_perm):
        orders[r] = rng.permutation(n)
    return orders


def relabel_assignments(n_subj: int, n_grp1: int, n_perm: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Two-group relabellings: (n_perm, n_subj) indices; columns ``[:n_grp1]`` are group 1.

    Exact (all ``C(n_subj, n_grp1)`` splits) when ``n_perm`` is at least that many; otherwise
    random orderings drawn with replacement, row 0 forced to the observed labelling.
    """
    if n_perm >= comb(n_subj, n_grp1):
        return enumerate_combinations(n_subj, n_grp1)
    return _random_orders(n_subj, n_perm, rng)


def row_permutations(n_subj: int, n_perm: int, rng: np.random.Generator) -> np.ndarray:
    """Row orderings for a correlation null: (n_perm, n_subj) indices, row 0 the identity.

    Exact (all ``n_subj!`` orderings, in MATLAB ``perms`` order with the identity swapped to the
    front) when ``n_subj <= 10`` and ``n_perm >= n_subj!``; otherwise random with replacement.
    """
    if n_subj <= 10 and n_perm >= factorial(n_subj):
        orders = np.array(list(permutations(range(n_subj))), dtype=int)[::-1].copy()
        orders[[0, -1]] = orders[[-1, 0]]
        return orders
    return _random_orders(n_subj, n_perm, rng)


def freedman_lane_indices(n_obs: int, n_perm: int, perm_type: str, blocks,
                          rng: np.random.Generator) -> np.ndarray:
    """Row indices for Freedman-Lane residual permutation: (n_perm, n_obs).

    ``perm_type='free'`` shuffles all rows; ``'within'`` shuffles rows only inside each
    exchangeability block (``blocks`` gives one block id per row). The observed ordering is not
    forced into the set, matching the GLM tier of the MATLAB toolbox.
    """
    out = np.empty((n_perm, n_obs), dtype=int)
    if perm_type == "free":
        for r in range(n_perm):
            out[r] = rng.permutation(n_obs)
        return out
    if perm_type != "within":
        raise ValueError(f"perm_type must be 'free' or 'within', got {perm_type!r}")
    blocks = np.asarray(blocks).ravel()
    if blocks.size != n_obs:
        raise ValueError("blocks must give one id per row")
    _, first = np.unique(blocks, return_index=True)
    groups = [np.flatnonzero(blocks == blocks[i]) for i in np.sort(first)]
    for r in range(n_perm):
        pe = np.arange(n_obs)
        for rows in groups:
            if rows.size > 1:
                pe[rows] = rows[rng.permutation(rows.size)]
        out[r] = pe
    return out
