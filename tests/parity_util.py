"""Helpers for the MATLAB parity tests (see parity_fixtures/README.md for the field dictionary)."""

from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from hdeeg_snpm.clusters import ChannelGraph

FIXTURES = Path(__file__).parent.parent / "parity_fixtures"
# the fixtures as they were before the cluster-statistic option (all built with the mean);
# copied by hand before the re-export, git-ignored, present only on the maintainer's machine
PRE_SUMMASS = Path(__file__).parent.parent / "_scratch" / "parity_fixtures_pre_summass"

STAT_RTOL = 1e-10
NULL_RTOL = 1e-9


def load(name: str, root: Path = FIXTURES) -> dict:
    path = root / f"{name}.mat"
    if not path.exists():
        pytest.skip(f"{path} is absent (built by scripts/export_parity_fixtures.m in MATLAB)")
    return sio.loadmat(path, simplify_cells=True)


def cluster_statistic(fx: dict) -> str:
    """The cluster-mass rule ('sum' or 'mean') MATLAB used for this fixture.

    MATLAB records it as ``results.cluster_statistic`` (and in ``params`` when it was set). A
    fixture without either predates the option, when every cluster mass was the mean, so it
    is replayed with 'mean'. Any other recorded value is an error, not a silent default.
    """
    for where in (fx.get("results"), fx.get("params")):
        value = where.get("cluster_statistic") if isinstance(where, dict) else None
        if isinstance(value, str) and value:      # an empty MATLAB char loads as an array
            rule = value.lower()
            assert rule in ("sum", "mean"), f"fixture records cluster_statistic {rule!r}"
            return rule
    return "mean"


def as_list(x) -> list:
    """A one-element MATLAB struct array loads as a single dict."""
    return [x] if isinstance(x, dict) else list(x)


def index0(x) -> np.ndarray:
    """1-based MATLAB indices (stored as small unsigned integers) to 0-based int64."""
    return np.atleast_1d(np.asarray(x)).astype(np.int64) - 1


def graph_of(fx: dict) -> ChannelGraph:
    nb = np.atleast_2d(np.asarray(fx["montage"]["neighbors"], dtype=float))
    return ChannelGraph(np.where(np.isnan(nb), -1, nb - 1).astype(np.int64))


def assert_close(actual, expected, rtol, what):
    """Largest absolute difference relative to the largest expected magnitude, NaN in one place."""
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    assert actual.shape == expected.shape, what
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), f"{what}: NaN pattern differs"
    if np.isnan(expected).all():
        return
    scale = np.nanmax(np.abs(expected))
    diff = np.nanmax(np.abs(actual - expected))
    assert diff <= rtol * max(scale, np.finfo(float).tiny), f"{what}: {diff / scale:.3g} relative"


def assert_p_equal(actual, expected, observed, null_sorted, what, two_sided=True,
                   identity_in_null=False):
    """Corrected p-values must be identical. A difference of one count is accepted only where the
    observed value lies within 1e-9 (relative) of a null value. Returns those channels. One count
    is 1/(N+1), or 1/N in the legacy tier (``identity_in_null``), where p = b/N."""
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    assert np.array_equal(np.isnan(actual), np.isnan(expected)), f"{what}: NaN pattern differs"
    null = np.asarray(null_sorted, dtype=float).ravel()
    null = null[~np.isnan(null)]
    obs = np.asarray(observed, dtype=float).ravel()
    obs = np.abs(obs) if two_sided else obs
    step = 1.0 / (np.asarray(null_sorted).size + (0 if identity_in_null else 1))
    ties = []
    for ch in np.flatnonzero(~np.isnan(expected) & (actual != expected)):
        gap = np.min(np.abs(null - obs[ch])) / max(abs(obs[ch]), np.finfo(float).tiny)
        assert abs(abs(actual[ch] - expected[ch]) - step) < 1e-12 and gap < 1e-9, (
            f"{what}: channel {ch} p {actual[ch]} vs {expected[ch]}, nearest null gap {gap:.3g}")
        ties.append(int(ch))
    return ties


def fwe_p(observed, null_sorted, two_sided):
    """MATLAB's corrected p from its own observed map and its own null."""
    obs = np.asarray(observed, dtype=float).ravel()
    ref = np.abs(obs) if two_sided else obs
    null = np.asarray(null_sorted, dtype=float).ravel()
    with np.errstate(invalid="ignore"):
        count = (null[None, :] >= ref[:, None]).sum(axis=1)
    return np.where(np.isnan(obs), np.nan, (count + 1) / (null.size + 1))


def matlab_has_2026_10_14_glm_fix(fx) -> bool:
    """Reference files written before the post-hoc and mixed2way changes of 2026-10-14 hold the
    old method for post-hoc loops and for the mixed2way group and condition effects. Files
    written by the updated export script say so in ``glm.scheme``."""
    return str(fx.get("glm", {}).get("scheme", "")) >= "2026-10-14"


def changed_2026_10_14(role: str) -> bool:
    """Loops whose method changed: every post-hoc, and the mixed2way condition effect."""
    return "/posthoc:" in role or role == "condition"
