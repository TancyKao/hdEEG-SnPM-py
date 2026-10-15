"""TFCE against the 206 maps stored by MATLAB's test_cluster_enhancement_identity.m."""

import time

import numpy as np
import pytest
import scipy.io as sio

from conftest import DATA
from hdeeg_snpm.clusters import find_clusters
from hdeeg_snpm.tfce import (
    DENSE_MAX_LEVELS,
    _n_levels,
    matlab_colon,
    tfce,
    tfce_event,
    tfce_reference,
)

# The two kinds of map with every |value| below the first threshold were stored wrong-sized until
# ClusterEnhancement.m was fixed on 2026-10-13 (sum along the threshold dimension); the fixture
# copied here is the corrected one, so every map is compared.


@pytest.fixture(scope="module")
def golden():
    return sio.loadmat(DATA / "ce_golden_178.mat", simplify_cells=True)


@pytest.mark.parametrize("fn", [tfce, tfce_reference], ids=["fast", "reference"])
def test_matches_matlab_golden_maps(golden, egi_graph, fn):
    checked = 0
    for values, kind, gold in zip(golden["maps"], golden["kind"], golden["tfce"]):
        out = fn(values, egi_graph, golden["E"], golden["H"])
        expected = np.asarray(gold, dtype=float)
        assert expected.shape == np.shape(values), kind
        checked += 1
        assert np.array_equal(np.isnan(out), np.isnan(expected)), kind
        np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0, equal_nan=True, err_msg=kind)
    assert checked == 206


def test_fast_equals_reference_on_random_maps(egi_graph):
    rng = np.random.default_rng(0)
    for scale, dh in [(3.0, 0.1), (25.0, 0.1), (0.2, 0.005)]:
        for _ in range(20):
            values = rng.standard_normal(egi_graph.n) * scale
            values[rng.random(egi_graph.n) < 0.05] = np.nan
            np.testing.assert_allclose(
                tfce(values, egi_graph, dh=dh), tfce_reference(values, egi_graph, dh=dh),
                rtol=1e-12, equal_nan=True)


def test_sign_symmetry_and_nonnegative_shortcut(egi_graph):
    rng = np.random.default_rng(1)
    values = rng.standard_normal(egi_graph.n) * 3
    np.testing.assert_allclose(tfce(-values, egi_graph), -tfce(values, egi_graph), rtol=1e-12)
    assert (tfce(np.abs(values), egi_graph) >= 0).all()


def test_all_nan_and_infinite(egi_graph):
    assert np.isnan(tfce(np.full(egi_graph.n, np.nan), egi_graph)).all()
    values = np.zeros(egi_graph.n)
    values[3] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        tfce(values, egi_graph)


def test_matlab_colon():
    np.testing.assert_allclose(matlab_colon(0.1, 0.1, 0.35), [0.1, 0.2, 0.3], rtol=1e-15)
    assert matlab_colon(0.1, 0.1, 0.1).tolist() == [0.1]
    assert matlab_colon(0.1, 0.1, np.nan).size == 0
    # 0.1:0.1:1.0 ends exactly on 1.0 in MATLAB
    assert matlab_colon(0.1, 0.1, 1.0)[-1] == 1.0 and matlab_colon(0.1, 0.1, 1.0).size == 10


# --------------------------------------------------------------------------- event path (round 4)

def _relative(a, b):
    return np.nanmax(np.abs(a - b)) / max(np.nanmax(np.abs(b)), 1e-300)


def test_event_path_agrees_with_dense_on_continuous_maps(egi_graph):
    rng = np.random.default_rng(7)
    smooth = egi_graph.dense + np.eye(egi_graph.n)
    smooth = smooth / smooth.sum(axis=1, keepdims=True)
    worst, checked = 0.0, 0
    for k in range(320):
        m = rng.standard_normal(egi_graph.n)
        for _ in range(rng.integers(0, 4)):
            m = smooth @ m
        # continuous: the largest value is a random multiple, not a level, so no value sits
        # exactly on a level (there the two paths' level values differ in the last bit)
        m *= rng.choice([0.3, 1, 3, 8, 30, 99]) * rng.uniform(0.5, 1) / np.abs(m).max()
        if k % 3 == 0:
            m = np.abs(m) ** 1.5                        # F-like, non-negative
        if k % 7 == 0:
            m[rng.choice(egi_graph.n, 5, replace=False)] = np.nan
        assert _n_levels(0.1, np.nanmax(np.abs(m))) <= DENSE_MAX_LEVELS
        worst = max(worst, _relative(tfce_event(m, egi_graph), tfce(m, egi_graph)))
        checked += 1
    print(f"event vs dense: {checked} maps, worst relative difference {worst:.2e}")
    assert checked >= 300 and worst < 1e-12, worst


def _reference_on_i_dh(values, graph, E=0.5, H=2, dh=0.1):
    """Direct sum over levels i * dh, the event path's own definition."""
    out = np.zeros(values.size)
    top = np.nanmax(np.abs(values))
    levels = np.arange(1, int(np.ceil(top / dh)) + 2) * dh
    for sign in (1, -1):
        side = np.where(sign * values < 0, 0.0, sign * values)
        for h in levels:
            for c in find_clusters(side, h, graph):
                out[c] += sign * c.size ** E * h ** H * dh
    out[np.isnan(values)] = np.nan
    return out


def test_event_path_on_level_values_matches_an_i_dh_reference(egi_graph):
    rng = np.random.default_rng(8)
    for k in range(40):
        m = rng.integers(-60, 61, egi_graph.n) * 0.1      # every value is some i * dh
        if k % 2:
            m = np.round(m)                              # whole numbers: on levels too
        if k % 5 == 0:
            m[rng.choice(egi_graph.n, 4, replace=False)] = np.nan
        np.testing.assert_allclose(tfce_event(m, egi_graph), _reference_on_i_dh(m, egi_graph),
                                   rtol=1e-12, atol=1e-12, equal_nan=True)


def test_huge_statistic_finishes_fast_and_is_exact(egi_graph):
    m = np.zeros(egi_graph.n)
    m[7] = 1e12
    start = time.perf_counter()
    out = tfce(m, egi_graph)                             # dispatches to the event path
    assert time.perf_counter() - start < 1.0
    n = 1e13 - 1                                         # levels i * 0.1 below 1e12
    np.testing.assert_allclose(out[7], 0.1 ** 3 * n * (n + 1) * (2 * n + 1) / 6, rtol=1e-12)
    assert (np.delete(out, 7) == 0).all()


@pytest.mark.parametrize("value", [1e15, 1e17, 1e20])
def test_statistic_beyond_double_level_resolution(egi_graph, value):
    """Above ~9e14 at dh 0.1, value / dh passes 2**53 and the level count cannot be corrected
    by one level; the event path must still finish (no error) and return v**3 / 3. Zero-variance
    rule 2 admits legitimate statistics this large (F up to about 1e20). MATLAB's
    ClusterEnhancement returns the same values bit for bit on these maps
    (_scratch/zero_variance/round4/event_large_check.m/.py)."""
    m = np.zeros(egi_graph.n)
    m[7] = value
    start = time.perf_counter()
    out = tfce(m, egi_graph)
    assert time.perf_counter() - start < 1.0
    np.testing.assert_allclose(out[7], value ** 3 / 3, rtol=1e-12)
    assert (np.delete(out, 7) == 0).all()
    rng = np.random.default_rng(5)
    mixed = rng.standard_normal(egi_graph.n) * 3
    hot = [7, *egi_graph.neighbor_lists[7]]
    mixed[hot] = value * (1 + 1e-3 * rng.random(len(hot)))
    mixed[90] = -value / 3
    mixed[91:94] = np.nan
    got = tfce(mixed, egi_graph)
    assert np.isfinite(got[~np.isnan(mixed)]).all() and np.isnan(got[91:94]).all()
    assert (got[hot] > 0).all() and got[90] < 0


def test_switch_point_and_event_path_limits(egi_graph):
    rng = np.random.default_rng(9)
    m = rng.standard_normal(egi_graph.n) * 3
    m[11] = 1500.0                                       # 15,001 levels: event path
    assert _n_levels(0.1, 1500.0) > DENSE_MAX_LEVELS
    np.testing.assert_allclose(tfce(m, egi_graph), tfce_event(m, egi_graph), rtol=0, atol=0)
    with pytest.raises(ValueError, match="whole-number H"):
        tfce_event(m, egi_graph, H=1.5)
    m[3] = -np.inf
    with pytest.raises(ValueError, match="infinite"):
        tfce(m, egi_graph)
