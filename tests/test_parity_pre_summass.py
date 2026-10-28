"""cluster_statistic='mean' reproduces the cluster masses MATLAB stored before the option existed.

The fixtures are the copy of ``parity_fixtures/`` taken before MATLAB re-exported them under the
'sum' default (``_scratch/parity_fixtures_pre_summass/``, git-ignored); every test skips when that
copy is absent. Under MATLAB's own permutation indices, 'mean' must give MATLAB's stored mass null
and cluster masses, 'sum' must give each observed cluster's mean mass times its size, and the TFCE
map, null and corrected p must be the same under both rules. The last test reads the re-exported fixtures instead and checks the
rule's wording against MATLAB's results text.
"""

import warnings

import numpy as np
import pytest

import parity_util as pu
from hdeeg_snpm import lmm
from hdeeg_snpm.circ_pipeline import run_circ
from hdeeg_snpm.glm import Design, run_effect

pytestmark = pytest.mark.parity


def _load(case):
    return pu.load(case, root=pu.PRE_SUMMASS)


def _glm_loop(fx, k, rule):
    loop = pu.as_list(fx["glm"]["loops"])[k]
    log = pu.as_list(fx["perm_log"])[k]
    C = np.asarray(loop["C"], dtype=float)
    design = Design(X=np.atleast_2d(np.asarray(loop["X"], dtype=float)),
                    C=C.reshape(1, -1) if C.ndim == 1 else C,
                    nuisance=np.asarray(loop["nuisance_idx"]).astype(bool),
                    blocks=np.asarray(loop["eb"]).astype(np.int64),
                    perm_type=loop["perm_type"], effect_label=loop["name"])
    return loop, run_effect(np.asarray(loop["Y"], dtype=float), design, pu.graph_of(fx),
                            perms=pu.index0(log["indices"]), alpha=float(fx["glm"]["alpha"]),
                            cluster_statistic=rule)


def _same_tfce(a, b, what):
    for f in ("real_T", "real_TFCE", "tMaxTFCE", "p_correctedTFCE", "p_real"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True), f"{what} {f}"


def _sum_is_mean_times_size(mean_clusters, sum_clusters, what):
    by_set = {frozenset(c.channels.tolist()): c.mass for c in mean_clusters if c.channels.size}
    assert len(by_set) == sum(c.channels.size > 0 for c in sum_clusters), what
    for c in sum_clusters:
        if c.channels.size:
            m = by_set[frozenset(c.channels.tolist())]
            assert c.mass == pytest.approx(m * c.channels.size, rel=1e-12), what


@pytest.mark.parametrize("case", ["rmanova", "mixed2way_2x3", "anova1"])
def test_glm_mean_reproduces_the_stored_masses(case):
    fx = _load(case)
    assert pu.cluster_statistic(fx) == "mean"
    for k, _ in enumerate(pu.as_list(fx["glm"]["loops"])):
        loop, mean = _glm_loop(fx, k, "mean")
        _, summed = _glm_loop(fx, k, "sum")
        what = f"{case} loop {k + 1} ({loop['name']})"
        pu.assert_close(mean.maxMass, loop["cluster_mass_null_sorted"], pu.NULL_RTOL,
                        f"{what} mean mass null")
        _same_tfce(mean, summed, what)
        _sum_is_mean_times_size(mean.clusters, summed.clusters, what)


def test_circular_mean_reproduces_the_stored_masses():
    from test_parity_circ import _params, _perms

    fx = _load("circ_phase_group")
    params = _params(fx)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for rule in ("mean", "sum"):
            out[rule] = run_circ({**params, "cluster_statistic": rule}, perms=_perms(fx)).result
    pu.assert_close(out["mean"].maxMass, fx["circ"]["cluster_mass_null_sorted"], pu.NULL_RTOL,
                    "circular mean mass null")
    _same_tfce(out["mean"], out["sum"], "circular")
    _sum_is_mean_times_size(out["mean"].clusters, out["sum"].clusters, "circular")


@pytest.mark.parametrize("case", ["lmm_factor", "lmm_continuous"])
def test_mixed_model_mean_reproduces_the_stored_masses(case):
    fx = _load(case)
    L = fx["lmm"]
    ev = np.asarray(L["evaluable"], dtype=bool).ravel()

    def correct(rule):
        maps = zip(np.asarray(L["perm_cluster_p"]), np.asarray(L["perm_cluster_wald"]))
        return lmm.cluster_correction(L["p_uncorrected"], L["wald"], maps, pu.graph_of(fx), ev,
                                      float(L["alpha"]), rule)

    (mean_c, mean_null), (sum_c, _) = correct("mean"), correct("sum")
    pu.assert_close(mean_null, L["cluster_mass_null_sorted"], pu.NULL_RTOL, f"{case} mass null")
    _sum_is_mean_times_size(mean_c, sum_c, case)


GLM_CIRC_LMM = ["anova1", "ancova_2group", "ancova_3group", "regression", "rmanova",
                "compu_anova1", "mixed2way_2x2", "mixed2way_2x3", "circ_phase_group",
                "circ_phase_group_u2", "circ_corrAngLinear", "lmm_continuous", "lmm_factor"]


@pytest.mark.parametrize("case", GLM_CIRC_LMM)
def test_rule_wording_matches_matlabs_results_text(case):
    """Re-exported fixtures (``parity_fixtures/``, not the backup): MATLAB's results text holds
    exactly the line Python writes, and the circular record is that line without its prefix."""
    from hdeeg_snpm import cluster_stat

    fx = pu.load(case)
    if "cluster_statistic" not in fx["results"]:
        pytest.skip(f"{case} predates the cluster_statistic option")
    rule = pu.cluster_statistic(fx)
    lines = [str(s) for s in np.atleast_1d(fx["results_text"]) if "Cluster statistic" in str(s)]
    is_circ = fx["tier"] == "circ"
    expected = (cluster_stat.circ_results_line if is_circ else cluster_stat.results_line)(rule)
    assert lines == [expected], case
    if is_circ:
        assert fx["results"]["circ"]["cluster_statistic"] == cluster_stat.circ_recorded(rule)
        assert fx["circ"]["cluster_statistic_rule"] == rule
    if fx["tier"] == "lmm":
        assert fx["lmm"]["cluster_statistic_rule"] == rule
