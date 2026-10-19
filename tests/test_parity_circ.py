"""Circular tier against MATLAB, end to end from the input files.

The Python pipeline reads the files MATLAB read (angles, counts or measure), is handed the row
indices of MATLAB's one permutation loop, and must reproduce every stage ``core_snpm_circ``
recorded: import, count rule, covariate, gate and channel set, convention check, Freedman-Lane
context, observed maps, TFCE map, sorted nulls, corrected p, clusters and the descriptive panel.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest

import parity_util as pu
from hdeeg_snpm import circ
from hdeeg_snpm.circ_pipeline import EXCLUSION_REASON, effect_label, run_circ

pytestmark = pytest.mark.parity

CASES = ["circ_phase_group", "circ_phase_group_u2", "circ_corrAngLinear", "circ_phase_group_gate",
         "circ_phase_group_u2_gate", "circ_phase_group_deg_luna",
         "circ_phase_group_u2_turtlewave", "circ_corrAngLinear_miss_y"]

_FILE_KEYS = ("data1_file", "data2_file", "count1_file", "count2_file")


def _params(fx):
    p = fx["params"]
    out = {k: p[k] for k in p if k not in ("output_path",) + _FILE_KEYS}
    for key in _FILE_KEYS:
        src = fx["source_files"].get(key)
        if isinstance(src, dict) and np.size(src.get("absolute")):
            if not Path(src["absolute"]).exists():
                pytest.skip(f"input file {src['absolute']} is not on this machine")
            out[key] = src["absolute"]
    return out


def _perms(fx):
    log = fx["perm_log"]
    assert isinstance(log, dict), "the circular tier has exactly one permutation loop"
    assert log["name"] == f"circ:{fx['params']['comparison']}"
    return np.atleast_2d(pu.index0(log["indices"]).reshape(np.shape(log["indices"])))


def _bool(x):
    return np.asarray(x).astype(bool)


def _text(x):
    """An empty MATLAB char loads as an empty array."""
    return x if isinstance(x, str) else ""


_RUNS: dict = {}


def run(case):
    if case not in _RUNS:
        fx = pu.load(case)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RUNS[case] = fx, run_circ(_params(fx), perms=_perms(fx))
    return _RUNS[case]


@pytest.mark.parametrize("case", CASES)
def test_inputs_and_channel_set(case):
    fx, out = run(case)
    c, res, pr = fx["circ"], fx["results"], out.prepared
    assert bool(c["replay_ok"]), f"{case}: MATLAB's own replay failed"
    assert list(pr.setup.labels) == list(fx["montage"]["labels"])
    assert np.array_equal(pr.setup.data_columns, pu.index0(fx["montage"]["data_columns"]))

    imp = c["import"]
    for k in ("units", "convention", "label"):
        assert pr.imp[k] == imp[k], k
    for k in ("rotation_deg", "resolution_deg", "n_finite"):
        assert pr.imp[k] == float(imp[k]), k
    assert pr.imp["quantised"] == bool(imp["quantised"])
    assert pr.imp["whole_degree"] == bool(imp["whole_degree"])

    pu.assert_close(pr.A_imported, c["angles_imported"], 0, "imported angles")
    pu.assert_close(pr.A, c["angles"], 0, "angles after the count rule")
    assert list(pr.subjects) == list(np.atleast_1d(c["subject_ids"]))
    if pr.is_group:
        assert np.array_equal(pr.gi, np.asarray(c["group"], dtype=float))
        pu.assert_close(pr.counts, c["counts"], 0, "counts")
        pu.assert_close(pr.logC, c["log_count"], 1e-14, "log count (libm, one ulp)")
        assert np.array_equal(pr.n_group1, np.asarray(c["n_group1"]).astype(int))
        assert np.array_equal(pr.n_group2, np.asarray(c["n_group2"]).astype(int))
        assert (pr.n1g, pr.n2g) == (int(res["circ"]["n_group1"]), int(res["circ"]["n_group2"]))
    else:
        assert pr.gi is None and pr.measure_col == c["measure_col"]
        pu.assert_close(pr.y, c["y"], 0, "measure")
        # subjects without a finite measure value are dropped up front, ids and count recorded
        rc = res["circ"]
        assert pr.measure_nan_removed == int(rc["measure_nan_removed"]
                                             if np.size(rc["measure_nan_removed"]) else 0)
        assert pr.dropped_subjects == [str(x) for x in np.atleast_1d(rc["dropped_subjects"])]
    assert np.array_equal(pr.n_finite, np.asarray(c["n_finite"]).astype(int))
    assert np.array_equal(pr.per_channel_n, np.asarray(res["per_channel_n"]).astype(int))

    for k in ("usecov", "collinear", "gate0", "gate", "complete", "evaluable"):
        assert np.array_equal(getattr(pr, k), _bool(c[k])), k
    assert np.array_equal(pr.evaluable, _bool(res["circ"]["evaluable"]))

    # excluded channels: the set, MATLAB's one combined reason, and one reason per route
    excl = res["excluded_channels"]
    assert pr.excluded.tolist() == (pu.index0(excl["index"]).tolist()
                                    if np.size(excl["index"]) else [])
    assert excl["reason"] == EXCLUSION_REASON
    expected = [_text(r) for r in np.atleast_1d(c["excluded_reason"])]
    assert [expected[i] for i in pr.excluded] == pr.excluded_reasons
    assert all(expected[i] == "" for i in np.flatnonzero(pr.evaluable))
    assert out.result.excluded_channels.tolist() == pr.excluded.tolist()


@pytest.mark.parametrize("case", CASES)
def test_covariate_and_convention_check(case):
    fx, out = run(case)
    c, pr = fx["circ"], out.prepared
    cov = c["covariate"]
    assert pr.covariate["mode"] == cov["mode"]
    for k in ("n_dropped_constant", "n_collinear", "n_cells_no_count"):
        assert pr.covariate[k] == int(cov[k]), k
    for k in ("pb_max", "pb_median", "frac_cells_no_count"):
        pu.assert_close(pr.covariate[k], cov[k], 1e-12, k)
    if "pb" in cov:
        pu.assert_close(pr.covariate["pb"], cov["pb"], 1e-12, "point-biserial r")
        assert pr.covariate["collinearity_max"] == float(cov["collinearity_max"])

    inv = c["inversion"]
    assert pr.inversion["verdict"] == inv["verdict"]
    assert pr.inversion["channels"].tolist() == pu.index0(inv["channels"]).tolist()
    assert pr.inversion["n_angles"] == int(inv["n_angles"])
    pu.assert_close(pr.inversion["mean_deg"], inv["mean_deg"], 1e-12, "pooled mean direction")
    pu.assert_close(pr.inversion["deviation_deg"], inv["deviation_deg"], 1e-12, "deviation")

    probe = c["inversion_probe"]
    if isinstance(probe, dict):
        params = dict(_params(fx), circ_convention=probe["convention"])
        with warnings.catch_warnings(), pytest.raises(circ.CircInputError) as err:
            warnings.simplefilter("ignore")
            run_circ(params, perms=_perms(fx))
        assert "core_snpm:" + err.value.code == probe["error_id"]
        assert str(err.value) == probe["message"]


@pytest.mark.parametrize("case", CASES)
def test_observed_maps_nulls_and_p(case):
    fx, out = run(case)
    c, res, r, eng = fx["circ"], fx["results"], out.result, out.engine
    assert eng.perm_type == c["perm_type"]
    pu.assert_close(out.prepared.tfce_dh, c["tfce_dh"], 0, "TFCE step")
    if c["perm_type"] == "free":
        fit = eng.fl_fit if eng.fl_fit.shape[1] > 1 else eng.fl_fit[:, 0]
        resid = eng.fl_resid if eng.fl_resid.shape[1] > 1 else eng.fl_resid[:, 0]
        pu.assert_close(fit, c["fl_Zfit"], 1e-12, "Freedman-Lane fit")
        pu.assert_close(resid, c["fl_R"], 1e-12, "Freedman-Lane residual")

    pu.assert_close(r.real_T, c["stat"], pu.STAT_RTOL, "statistic")
    pu.assert_close(r.real_T, res["T"]["real_T"], pu.STAT_RTOL, "statistic (results)")
    pu.assert_close(r.p_real, c["p_uncorrected"], 1e-8, "uncorrected p")
    pu.assert_close(r.real_TFCE, c["tfce"], pu.NULL_RTOL, "TFCE map")
    pu.assert_close(r.tMaxTFCE, res["T"]["tMaxTFCE"], pu.NULL_RTOL, "max-TFCE null")
    pu.assert_close(r.tMaxTFCE, c["tfce_null_sorted"], pu.NULL_RTOL, "max-TFCE null (replay)")
    pu.assert_close(r.maxMass, c["cluster_mass_null_sorted"], pu.NULL_RTOL, "cluster-mass null")
    pu.assert_close(r.critical_T_TFCE, res["T"]["critical_T_TFCE"], pu.NULL_RTOL, "critical TFCE")
    ties = pu.assert_p_equal(r.p_correctedTFCE, res["p"]["correctedTFCE"], res["T"]["real_TFCE"],
                             res["T"]["tMaxTFCE"], "TFCE p", two_sided=False)
    assert not ties, f"{case}: p differs by one count at channels {ties}"
    if out.prepared.comparison == "circ_corrAngLinear":
        pu.assert_close(eng.rho, res["circ"]["rho"], pu.STAT_RTOL, "descriptive rho")


@pytest.mark.parametrize("case", CASES)
def test_clusters_and_significant_channels(case):
    fx, out = run(case)
    res = fx["results"]
    theirs = [cl for cl in pu.as_list(res["Clusters"]) if np.size(cl["channels"])]
    ours = [cl for cl in out.result.clusters if cl.channels.size]
    assert len(ours) == len(theirs)
    ref = {frozenset(pu.index0(cl["channels"]).tolist()): cl for cl in theirs}
    for cl in ours:
        m = ref[frozenset(cl.channels.tolist())]
        assert cl.p == float(m["p"])
        assert abs(cl.mass - float(m["mass"])) <= pu.NULL_RTOL * abs(float(m["mass"]))
        assert cl.threshold == float(m["threshold"]) and cl.permutations == int(m["permutations"])
    for name in ("uncorrsigch", "correctTFCEsigch", "SnPMsigch"):
        expected = sorted(pu.index0(res[name]).tolist()) if np.size(res[name]) else []
        assert sorted(getattr(out, name).tolist()) == expected, name


@pytest.mark.parametrize("case", CASES)
def test_descriptive_panel_and_labels(case):
    fx, out = run(case)
    res = fx["results"]
    d, ref = out.descriptive, res["circ_descriptive"]
    groups = pu.as_list(ref["group"])
    assert d["n_channels"] == int(ref["n_channels"]) and d["note"] == ref["note"]
    assert len(d["group"]) == len(groups)
    for ours, theirs in zip(d["group"], groups):
        assert ours["label"] == theirs["label"] and ours["n"] == int(theirs["n"])
        for k in ("mean_deg", "sd_deg", "R", "rayleigh_z", "rayleigh_p"):
            pu.assert_close(ours[k], theirs[k], 1e-10, f"descriptive {theirs['label']} {k}")
    assert effect_label(out.prepared) == res["glm"]["effect_label"]
    assert "global_stat" not in res and "global_pval" not in res
