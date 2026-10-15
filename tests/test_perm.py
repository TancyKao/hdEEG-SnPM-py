from math import comb, factorial

import numpy as np

from hdeeg_snpm import perm


def rng():
    return np.random.default_rng(0)


def test_signflip_exact_and_random():
    p = perm.signflip_patterns(3, 8, rng())
    assert p.shape == (8, 3) and p[0].tolist() == [False] * 3 and p[-1].all()
    assert p[1].tolist() == [True, False, False] and p[2].tolist() == [False, True, False]
    assert len({tuple(r) for r in p}) == 8
    q = perm.signflip_patterns(20, 500, rng())
    assert q.shape == (500, 20) and q[0].all()
    assert 0.4 < q[1:].mean(axis=0).min() and q[1:].mean(axis=0).max() < 0.6


def test_enumerate_combinations_order():
    c = perm.enumerate_combinations(4, 2)
    assert c.tolist() == [[0, 1, 2, 3], [0, 2, 1, 3], [0, 3, 1, 2],
                          [1, 2, 0, 3], [1, 3, 0, 2], [2, 3, 0, 1]]
    c = perm.enumerate_combinations(7, 3)
    assert c.shape == (comb(7, 3), 7) and (np.sort(c, axis=1) == np.arange(7)).all()


def test_relabel_and_row_permutations():
    assert perm.relabel_assignments(8, 4, 100, rng()).shape == (70, 8)
    r = perm.relabel_assignments(20, 10, 300, rng())
    assert r.shape == (300, 20) and r[0].tolist() == list(range(20))
    assert (np.sort(r, axis=1) == np.arange(20)).all()

    o = perm.row_permutations(3, 6, rng())
    # MATLAB perms(1:3) is [3 2 1; 3 1 2; 2 3 1; 2 1 3; 1 3 2; 1 2 3], identity swapped to row 1
    assert o.tolist() == [[0, 1, 2], [2, 0, 1], [1, 2, 0], [1, 0, 2], [0, 2, 1], [2, 1, 0]]
    assert perm.row_permutations(5, 500, rng()).shape == (factorial(5), 5)
    assert perm.row_permutations(11, 200, rng()).shape == (200, 11)


def test_freedman_lane_indices():
    free = perm.freedman_lane_indices(12, 50, "free", None, rng())
    assert (np.sort(free, axis=1) == np.arange(12)).all()
    blocks = np.array(["b", "a", "b", "a", "c", "b"])
    within = perm.freedman_lane_indices(6, 200, "within", blocks, rng())
    assert (blocks[within] == blocks).all(), "rows never leave their block"
    assert (within[:, 4] == 4).all(), "a single-row block is left alone"
    assert len({tuple(r) for r in within}) == 12  # 3! * 2! orderings, all reached
