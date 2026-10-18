"""t-test and correlation analyses against MATLAB, end to end from the input files.

The Python pipeline reads the same files MATLAB read, is handed MATLAB's permutation indices
for the TFCE loop and for the cluster loop, and must reproduce MATLAB's results.
"""

from pathlib import Path

import numpy as np
import pytest

import parity_util as pu
from hdeeg_snpm.pipeline import run_legacy

pytestmark = pytest.mark.parity

CASES = [
    "pairedT_both", "pairedT_left", "pairedT_right", "unpairedT_both", "unpairedT_left",
    "unpairedT_right", "onesampleT_single", "onesampleT_pair", "correlationP_wide",
    "correlationS_wide", "correlationP_single", "correlationS_single", "correlationP_wide_cov",
    "correlationS_wide_cov", "correlationP_single_cov", "correlationS_single_cov",
    "exact_pairedT_n6", "exact_unpairedT_4v4", "exact_correlationP_n5", "compu_pairedT",
    "compu_unpairedT", "compu_correlationP", "miss_pairedT_scattered",
    "miss_correlationP_scattered", "miss_unpairedT_group_channel", "miss_correlationP_cov_nan",
    "exact_correlationS_n7", "exact_correlationS_n7_ties", "correlationS_ties_n30",
    "pairedT_logscale", "correlationP_normalize",
]


# Channels where a corrected p may differ from MATLAB by exactly one count because the observed
# value is mathematically equal to a null value and MATLAB's rounding fell on the other side.
# Empty since round 4 (2026-10-14): MATLAB counts ties within 1e-12 relative as well
# (snpm_count_at_least.m), and the regenerated fixtures carry that rule. The one former entry,
# pairedT_logscale channel 78 (observed TFCE 5.525000000000001 against a null value of 5.525),
# now agrees.
KNOWN_TIES: dict[str, list[int]] = {}
# Per-subject z-scoring makes every subject's channel average zero, so the whole-head
# correlation is a correlation of rounding error. MATLAB flags it as not interpretable.
GLOBAL_NOT_INTERPRETABLE = {"correlationP_normalize"}


def _perms(log):
    idx = np.atleast_2d(np.asarray(log["indices"])).astype(np.int64)
    return idx if log["kind"] == "signflip" else idx - 1


def _params(fx):
    p = fx["params"]
    out = {k: p[k] for k in ("comparison", "tail", "channels", "datatype") if k in p}
    for key in ("data1_file", "data2_file", "covariate_file"):
        src = fx["source_files"].get(key)
        if isinstance(src, dict) and np.size(src.get("absolute")):
            if not Path(src["absolute"]).exists():
                pytest.skip(f"input file {src['absolute']} is not on this machine")
            out[key] = src["absolute"]
    out["use_covariates"] = bool(p.get("use_covariates", False))
    return out


def run(case):
    fx = pu.load(case)
    logs = pu.as_list(fx["perm_log"])
    assert [lg["name"] for lg in logs] == ["legacy/TFCE", "legacy/cluster"]
    return fx, run_legacy(_params(fx), perms_tfce=_perms(logs[0]), perms_cluster=_perms(logs[1]))


@pytest.mark.parametrize("case", CASES)
def test_matches_matlab(case):
    fx, out = run(case)
    res, T, p = fx["results"], fx["results"]["T"], fx["results"]["p"]
    two_sided = fx["params"]["tail"] == "both"
    prep = out.prepared

    assert list(prep.setup.labels) == list(fx["montage"]["labels"])
    excluded = res.get("excluded_channels", {})
    expected_excl = pu.index0(excluded["index"]) if np.size(excluded.get("index", [])) else []
    assert prep.rule.excluded.tolist() == list(expected_excl)
    assert np.array_equal(prep.n_analysed, np.asarray(res["per_channel_n"]).astype(int))
    assert np.array_equal(prep.rule.n_available, np.asarray(res["n_available"]).astype(int))

    # what MATLAB handed to its engines, after matching, selection, transform, masking and
    # covariate residualisation
    eng = fx["engine"]
    assert prep.stat_comparison == eng["comparison"] and prep.ncov == int(eng["ncov"])
    pu.assert_close(prep.x, eng["data_x"], 1e-10, "engine input x")
    pu.assert_close(prep.y, eng["data_y"], 1e-10, "engine input y")
    # Freedman-Lane context: present exactly for the covariate cases
    if "fl_Rz" in eng:
        assert prep.fl is not None and prep.fl.k_eff == int(eng["k_eff"])
        pu.assert_close(prep.fl.Rz, eng["fl_Rz"], 1e-10, "Freedman-Lane Rz")
    else:
        assert prep.fl is None
    pu.assert_close(prep.threshold, abs(float(eng["cluster_threshold"])), 1e-9, "threshold")
    assert np.array_equal(np.flatnonzero(out.global_.channels_kept),
                          pu.index0(eng["global_channels_kept"]))
    pu.assert_close(out.clusters.max_cluster_sizes, eng["cluster_null_sorted"], 0,
                    "cluster-size null")

    pu.assert_close(out.T.real_T, T["real_T"], pu.STAT_RTOL, "statistic")
    pu.assert_close(out.T.real_TFCE, T["real_TFCE"], pu.NULL_RTOL, "TFCE map")
    pu.assert_close(out.T.tMax, T["tMax"], pu.NULL_RTOL, "max-statistic null")
    pu.assert_close(out.T.tMaxTFCE, T["tMaxTFCE"], pu.NULL_RTOL, "max-TFCE null")
    pu.assert_close(out.T.chk_T, T["chk_T"], pu.NULL_RTOL, "mean permuted statistic")
    pu.assert_close(out.T.critical_T, T["critical_T"], pu.NULL_RTOL, "critical statistic")
    pu.assert_close(out.T.critical_T_TFCE, T["critical_T_TFCE"], pu.NULL_RTOL, "critical TFCE")
    pu.assert_close(out.T.p_real, p["real"], 1e-8, "uncorrected p")
    ties = pu.assert_p_equal(out.T.p_corrected, p["corrected"], T["real_T"], T["tMax"],
                             "max-statistic p", two_sided)
    ties += pu.assert_p_equal(out.T.p_correctedTFCE, p["correctedTFCE"], T["real_TFCE"],
                              T["tMaxTFCE"], "TFCE p", two_sided)
    assert ties == KNOWN_TIES.get(case, []), f"p differs by one count at channels {ties}"

    def key(channels, threshold):
        return (round(float(threshold), 9), frozenset(np.asarray(channels).tolist()))

    theirs = {key(pu.index0(c["channels"]) if np.size(c["channels"]) else [], c["threshold"]):
              float(c["p"]) for c in pu.as_list(res["Clusters"])}
    ours = {key(c.channels, c.threshold): c.p for c in out.clusters.clusters}
    assert ours == theirs

    assert out.global_.n_channels == int(res["global_n_channels"])
    if case not in GLOBAL_NOT_INTERPRETABLE:
        pu.assert_close(out.global_.stat, res["global_stat"], 1e-10, "whole-head statistic")
        pu.assert_close(out.global_.pval, res["global_pval"], 1e-8, "whole-head p")
    else:
        assert not prep.global_interpretable
    for name in ("uncorrsigch", "correctTFCEsigch", "SnPMsigch"):
        assert sorted(getattr(out, name).tolist()) == sorted(pu.index0(res[name]).tolist()) \
            if np.size(res[name]) else getattr(out, name).size == 0, name
