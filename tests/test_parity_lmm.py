"""Mixed linear model tier against MATLAB.

The fit is compared by tolerance: fitlme stops its optimiser at a gradient of about 1e-6, so its
numbers carry that error and cannot be matched to rounding. Measured on the four fixtures
(_scratch/lmm_bench/compare_fixtures.py): worst element-wise relative difference 3.6e-7 on the
statistic, 1.7e-6 on the model p, 2.2e-7 on the subject SD, 2e-14 on the log-likelihood. The
correction is compared exactly, by feeding MATLAB's recorded permuted maps to the Python TFCE and
cluster code.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import parity_util as pu
from hdeeg_snpm import lmm, lmm_pipeline
from hdeeg_snpm.io import tables

pytestmark = pytest.mark.parity

CASES = ["lmm_continuous", "lmm_factor", "lmm_missing", "lmm_continuous_groupcov",
         "lmm_meta_missing", "lmm_degenerate"]


def _degenerate(fx, n):
    """MATLAB's lmm.degenerate: channels with no residual variation (logical mask)."""
    d = fx["lmm"].get("degenerate")
    return np.zeros(n, bool) if d is None or not np.size(d) else np.asarray(d, bool).ravel()

FIT_RTOL = 1e-6      # largest difference relative to the largest magnitude in the map
LOGLIK_ATOL = 1e-8

# Permuted fits (0-based draws per loop: TFCE, cluster) where MATLAB's fitlme stopped short of the
# likelihood maximum, so its recorded statistic differs from Python's by more than the fit
# tolerance while Python's log-likelihood at that channel is higher. In lmm_degenerate, TFCE 161
# and 188 are theta = 0 stalls whose likelihood gain (9e-8, 3e-8) is below the 1e-6 that MATLAB's
# refit guard needs; the others are interior points (log-likelihood short by 2e-6 to 1.2e-4).
# Every draw of both loops is refitted, so a new shortfall shows up as a failure here.
# Every draw of both loops (200 + 200) is refitted for the two cases where MATLAB's optimiser
# has fallen short (lmm_degenerate, lmm_factor); the others refit their first 40 draws per loop to
# keep the suite fast (all draws of all six cases take about 200 s). Set SNPM_PARITY_ALL_DRAWS=1
# to refit every draw everywhere; that run passed with the record below on 2026-10-20.
ALL_DRAWS = {"lmm_degenerate", "lmm_factor"}
REFIT_FIRST = 40
KNOWN_MATLAB_OPTIMISER_SHORTFALLS: dict[str, tuple[set, set]] = {
    "lmm_degenerate": ({66, 93, 142, 161, 181, 188}, {128, 154, 188}),
}


def _model(fx):
    L = fx["lmm"]
    meta = pd.DataFrame({k: np.asarray(v).ravel() for k, v in L["meta"].items()})
    spec = lmm.LMMSpec(dv=L["dv"], fixed=L["fixed"], random=L["random"], effect=L["effect"],
                       effect_type=L["effect_type"], subject=L["subject_col"],
                       group=L["group_col"])
    assert spec.perm == L["perm_scheme"]
    return lmm.LMM(meta, spec), np.asarray(L["power"], dtype=float), \
        np.asarray(L["evaluable"], dtype=bool).ravel()


@pytest.mark.parametrize("case", CASES)
def test_observed_fit_matches_fitlme(case):
    fx = pu.load(case)
    L = fx["lmm"]
    assert L["replay_ok"]
    model, power, ev = _model(fx)
    assert model.names == list(np.atleast_1d(L["coef_names"]))
    assert model.effect_cols == list(pu.index0(L["effect_coef_rows"]))
    const = (np.asarray(L["const_within_subject"], bool).ravel() if "const_within_subject" in L
             else np.zeros(ev.size, bool))
    assert np.array_equal(ev, np.isfinite(power).all(axis=0) & ~const)
    fit = model.fit(power, evaluable=ev)
    dg = _degenerate(fx, ev.size)
    assert np.array_equal(np.isfinite(fit.stat), ev & ~dg)
    assert (fit.wald[dg] == 0).all() and np.isnan(fit.p_model[dg]).all()
    X, y = model.design(power[:, ev], model.meta)
    assert np.array_equal(lmm.degenerate(y, X, model.gidx, model.n_groups), dg[ev])
    pu.assert_close(fit.stat, L["stat"], FIT_RTOL, f"{case} statistic")
    pu.assert_close(fit.wald, L["wald"], FIT_RTOL, f"{case} wald")
    pu.assert_close(fit.p_model, L["p_uncorrected"], FIT_RTOL, f"{case} model p")
    assert np.nanmax(np.abs(fit.p_model - L["p_uncorrected"]) / L["p_uncorrected"]) < 1e-5
    for name, mine, ref in [("estimate", fit.beta.T, L["coef_estimate"]),
                            ("SE", fit.se.T, L["coef_se"]),
                            ("t", (fit.beta / fit.se).T, L["coef_t"])]:
        for i, coef in enumerate(model.names):
            pu.assert_close(mine[i], np.atleast_2d(ref)[i], FIT_RTOL, f"{case} {name} {coef}")
    assert (np.asarray(L["df2"]).ravel()[ev & ~dg] == fit.df).all()
    assert np.nanmax(np.abs(fit.loglik - L["loglik"])) < LOGLIK_ATOL
    pu.assert_close(fit.sd_subject, L["re_std"], FIT_RTOL, f"{case} subject SD")
    pu.assert_close(fit.sigma, L["residual_std"], FIT_RTOL, f"{case} residual SD")
    excluded = pu.index0(fx["results"]["T"]["excluded_channels"]) if np.size(
        fx["results"]["T"]["excluded_channels"]) else np.empty(0, int)
    assert np.array_equal(np.flatnonzero(~ev), excluded)


@pytest.mark.parametrize("case", CASES)
def test_correction_from_matlab_permuted_maps_is_exact(case):
    fx = pu.load(case)
    L = fx["lmm"]
    g = pu.graph_of(fx)
    ev = np.asarray(L["evaluable"], dtype=bool).ravel()
    t = lmm.tfce_correction(L["stat"], L["p_uncorrected"], np.asarray(L["perm_stat"]), g, ev,
                            L["tail"], float(L["alpha"]), float(L["tfce_E"]), float(L["tfce_H"]))
    pu.assert_close(t["real_TFCE"], L["tfce"], pu.NULL_RTOL, f"{case} TFCE map")
    pu.assert_close(t["tMaxTFCE"], L["tfce_null_sorted"], pu.NULL_RTOL, f"{case} TFCE null")
    assert np.array_equal(t["p_correctedTFCE"], np.asarray(L["p_correctedTFCE"]).ravel(),
                          equal_nan=True)
    assert np.array_equal(t["p_correctedTFCE"],
                          np.asarray(fx["results"]["p"]["correctedTFCE"]).ravel(), equal_nan=True)
    clusters, mass = lmm.cluster_correction(
        L["p_uncorrected"], L["wald"],
        zip(np.asarray(L["perm_cluster_p"]), np.asarray(L["perm_cluster_wald"])), g, ev,
        float(L["alpha"]))
    pu.assert_close(mass, L["cluster_mass_null_sorted"], pu.NULL_RTOL, f"{case} mass null")
    mine = sorted((tuple(sorted(c.channels.tolist())), c.p) for c in clusters)
    ref = []
    for c in pu.as_list(fx["results"]["Clusters"]):
        ch = tuple(sorted(pu.index0(c["channels"]).tolist())) if np.size(c["channels"]) else ()
        ref.append((ch, float(c["p"])))
    assert mine == sorted(ref)
    for c in clusters:
        match = [r for r in pu.as_list(fx["results"]["Clusters"]) if np.size(r["channels"]) and
                 sorted(pu.index0(r["channels"]).tolist()) == sorted(c.channels.tolist())]
        if match:
            assert c.mass == pytest.approx(float(match[0]["mass"]), rel=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_refits_under_matlab_indices_match_the_recorded_maps(case):
    """Draws are refitted from MATLAB's indices (see ALL_DRAWS). A channel may differ from
    MATLAB's recorded value by more than the fit tolerance only where MATLAB's optimiser stopped
    short: the variance ratio implied by MATLAB's value has a lower likelihood than Python's."""
    fx = pu.load(case)
    L = fx["lmm"]
    model, power, ev = _model(fx)
    logs = pu.as_list(fx["perm_log"])
    known = KNOWN_MATLAB_OPTIMISER_SHORTFALLS.get(case, (set(), set()))
    dg = _degenerate(fx, ev.size)
    for loop, key in [(0, "perm_stat"), (1, "perm_cluster_wald")]:
        idx = pu.index0(logs[loop]["indices"])
        recorded = np.asarray(L[key], dtype=float)
        found = set()
        every = case in ALL_DRAWS or os.environ.get("SNPM_PARITY_ALL_DRAWS") == "1"
        draws = range(idx.shape[0]) if every else range(min(REFIT_FIRST, idx.shape[0]))
        for i in draws:
            meta = lmm.permute_meta(model.meta, model.spec, idx[i])
            fit = model.fit(power, meta, ev)
            mine = fit.stat if loop == 0 else fit.wald
            mine = np.where(ev, mine, np.nan if loop == 0 else 0.0)
            if loop == 0:
                assert np.array_equal(np.isnan(mine), np.isnan(recorded[i])), f"{case} draw {i}"
            else:
                p_rec = np.asarray(L["perm_cluster_p"], dtype=float)[i]
                assert np.array_equal(np.isnan(np.where(ev, fit.p_model, np.nan)),
                                      np.isnan(p_rec)), f"{case} draw {i}"
            assert np.isnan(fit.stat[dg]).all() and (fit.wald[dg] == 0).all()
            scale = np.nanmax(np.abs(recorded[i]))
            bad = np.flatnonzero(np.abs(mine - recorded[i]) > FIT_RTOL * scale)
            if bad.size:
                found.add(i)
                X, y = model.design(power[:, bad], meta)
                s = lmm._sufficient(y, X, model.gidx, model.n_groups)
                squared = loop == 1 and model.spec.effect_type == "continuous"
                for k, ch in enumerate(bad):
                    th_m = _implied_theta(s, k, model, recorded[i, ch], squared, fit.theta[ch])
                    assert th_m is not None, f"{case} loop {loop} draw {i} ch {ch}: no ratio"
                    dev_py = _dev_one(s, k, fit.theta[ch])
                    dev_ml = _dev_one(s, k, th_m)
                    assert dev_py <= dev_ml + 1e-9, (
                        f"{case} loop {loop} draw {i} ch {ch}: MATLAB likelihood is higher")
        expected = {d for d in known[loop] if d in set(draws)}
        assert found == expected, f"{case} loop {loop}: {sorted(found)}"


def _one(s, k):
    return {key: (v[k:k + 1] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] ==
                  s["Sx"].shape[0] else v) for key, v in s.items()}


def _dev_one(s, k, theta):
    return float(lmm._deviance(_one(s, k), np.array([theta]))[0])


def _stat_at(s, model, theta):
    """The effect statistic of the GLS fit at a given variance ratio, every channel of s."""
    XtWX, beta, r2, _ = lmm._gls(s, np.asarray(theta, dtype=float))
    cov = (r2 / s["N"])[:, None, None] * np.linalg.inv(XtWX)
    e = model.effect_cols
    if model.spec.effect_type == "continuous":
        return beta[:, e[0]] / np.sqrt(cov[:, e[0], e[0]])
    b = beta[:, e]
    return np.einsum("cp,cp->c", b, np.linalg.solve(cov[:, e][:, :, e], b[..., None])[..., 0]) \
        / len(e)


def _implied_theta(s, k, model, value, squared, theta_py):
    """The variance ratio at which the GLS statistic equals MATLAB's recorded value (the root
    nearest Python's optimum), i.e. where MATLAB's optimiser stopped."""
    from scipy.optimize import brentq

    one = _one(s, k)

    def f(t):
        v = _stat_at(one, model, np.array([t]))[0]
        return (v * v if squared else v) - value

    hi = max(10.0, 10 * theta_py)
    grid = np.concatenate([[0.0], np.linspace(1e-6, hi, 4001)])
    vals = np.array([f(t) for t in grid])
    if abs(vals[0]) <= 1e-7 * abs(value):
        roots = [0.0]
    else:
        roots = []
    for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if np.sign(fa) != np.sign(fb):
            roots.append(brentq(f, a, b, xtol=1e-14))
    if not roots:
        return None
    return min(roots, key=lambda r: abs(r - theta_py))


def _params(fx):
    params = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in fx["params"].items()}
    if not Path(params["data_file"]).exists():
        pytest.skip(f"{params['data_file']} is absent (MATLAB repository not on this machine)")
    return params


@pytest.mark.parametrize("case", CASES)
def test_pipeline_reads_the_same_rows_channels_and_whole_head_test(case):
    fx = pu.load(case)
    L = fx["lmm"]
    params = _params(fx)
    if int(L["dropped_rows"]["n"]):
        with pytest.warns(lmm.LMMMetaNaNRows, match=r"33 of 300 rows \(trials\) dropped for a "
                          r"missing model variable \(group 3, sleepDepth 30\); subjects that "
                          "lost every trial: none"):
            spec, meta, power, setup, ev, dropped, _, reasons, w_ch = lmm_pipeline.prepare(params)
    else:
        spec, meta, power, setup, ev, dropped, _, reasons, w_ch = lmm_pipeline.prepare(params)
    assert list(setup.labels) == list(fx["montage"]["labels"])
    assert np.array_equal(power, np.asarray(L["power"], dtype=float), equal_nan=True)
    assert np.array_equal(ev, np.asarray(L["evaluable"], dtype=bool).ravel())
    assert spec.perm == L["perm_scheme"]
    d = L["dropped_rows"]
    assert dropped.n == int(d["n"]) and dropped.n_input == int(d["n_input"])
    assert list(dropped.rows) == list(pu.index0(d["rows"])) if int(d["n"]) else dropped.n == 0
    assert dropped.columns == list(np.atleast_1d(d["columns"]))
    assert dropped.per_column == [int(v) for v in np.atleast_1d(d["per_column"])]
    assert dropped.subjects_lost == list(np.atleast_1d(d["subjects_lost"]))
    assert power.shape[0] == int(fx["results"]["lmm"]["n_trials"])
    assert dropped.n_subjects == int(fx["results"]["lmm"]["n_subjects"])
    for col in ("Subject", "sleepDepth"):
        ref = np.asarray(L["meta"][col]).ravel()
        assert list(meta[col]) == list(ref), col
    # excluded channels and their reasons, in channel order, as MATLAB's record
    xc = fx["results"]["excluded_channels"]
    idx = pu.index0(xc["index"]) if np.size(xc["index"]) else np.empty(0, int)
    assert sorted(reasons) == list(idx)
    want = (list(np.atleast_1d(xc["reasons"])) if "reasons" in xc and np.size(xc["reasons"])
            else [str(xc["reason"])] * idx.size)
    assert [reasons[i] for i in sorted(reasons)] == want
    line = lmm_pipeline.excluded_text(spec, setup.labels, sorted(reasons),
                                      [reasons[i] for i in sorted(reasons)], ev.size)
    assert line in [str(t) for t in fx["results_text"]]
    if "within_ratio" in L:
        ref = np.asarray(L["within_ratio"], dtype=float).ravel()
        assert np.array_equal(np.isnan(w_ch), np.isnan(ref))
        assert np.allclose(w_ch, ref, rtol=1e-9, atol=1e-15, equal_nan=True)
        const = np.asarray(L["const_within_subject"], bool).ravel()
        # MATLAB's exclusion list also holds other reasons (e.g. incomplete channels in
        # lmm_missing); the constant-within-subject flags must match exactly the entries
        # excluded for that reason
        xi = pu.index0(np.atleast_1d(L["excluded_index"])) if np.size(L["excluded_index"]) else []
        xr = [str(r) for r in np.atleast_1d(L["excluded_reasons"])] if len(xi) else []
        const_x = [int(i) for i, r in zip(xi, xr) if r in (lmm.CONST_RESPONSE, lmm.CONST_PREDICTOR)]
        assert np.flatnonzero(const).tolist() == const_x
    gstat, gp = lmm_pipeline.global_test(meta, power, spec, ev)
    assert gstat == pytest.approx(float(L["global_stat"]), rel=1e-6)
    assert gp == pytest.approx(float(L["global_p"]), rel=1e-5)


def test_pipeline_end_to_end_under_matlab_indices():
    """lmm_continuous through run_lmm with MATLAB's two index matrices: the TFCE null, the
    corrected p and the clusters come out as MATLAB's (no boundary stall in this case)."""
    fx = pu.load("lmm_continuous")
    params = _params(fx)
    logs = pu.as_list(fx["perm_log"])
    res = lmm_pipeline.run_lmm(params, perms_tfce=pu.index0(logs[0]["indices"]),
                               perms_cluster=pu.index0(logs[1]["indices"]))
    L, R = fx["lmm"], fx["results"]
    pu.assert_close(res.T.tMaxTFCE, L["tfce_null_sorted"], FIT_RTOL, "TFCE null")
    pu.assert_close(res.clusters.maxMass, L["cluster_mass_null_sorted"], FIT_RTOL, "mass null")
    ties = pu.assert_p_equal(res.T.p_correctedTFCE, np.asarray(R["p"]["correctedTFCE"]).ravel(),
                             L["tfce"], L["tfce_null_sorted"], "TFCE p")
    assert ties == []
    mine = sorted((tuple(sorted(c.channels.tolist())), c.p) for c in res.clusters.clusters)
    ref = sorted((tuple(sorted(pu.index0(c["channels"]).tolist())), float(c["p"]))
                 for c in pu.as_list(R["Clusters"]))
    assert mine == ref



def test_backstop_keep_drop_decisions_match_matlab():
    """lmm_backstop_sweep: y ~ group + (1|Subject), 12 subjects x 5 trials, with the method-of-
    moments subject/residual SD ratio set to 1e2, 5e2, 999, 1001, 2e3, 1e4. degenerate() must
    reproduce MATLAB's rss_x, rss_xz, theta and decisions (0 0 0 1 1 1 at the 1e3 cutoff), and
    within_ratio its within-subject shares."""
    data, expected = (pu.FIXTURES / "lmm_backstop_sweep_data.csv",
                      pu.FIXTURES / "lmm_backstop_sweep_expected.csv")
    if not data.exists():
        pytest.skip(f"{data} is absent (built by scripts/export_lmm_backstop_sweep.m)")
    d = tables.read_csv(data)
    e = tables.read_csv(expected)
    model = lmm.LMM(d.assign(POWER=0.0), lmm.LMMSpec(dv="POWER", fixed="group", effect="group",
                                                      effect_type="factor"))
    cols = list(e["column"])
    Y = d[cols].to_numpy(float)
    X, _ = model.design(Y)
    info = lmm.degenerate_info(Y, X, model.gidx, model.n_groups)
    assert list(info["degenerate"].astype(int)) == list(e["degenerate"]) == [0, 0, 0, 1, 1, 1]
    assert np.allclose(np.sqrt(info["theta2"]), e["theta_mom"], rtol=1e-9)
    assert np.allclose(info["rss_x"], e["rss_x"], rtol=1e-9)
    assert np.allclose(info["rss_xz"], e["rss_xz"], rtol=1e-6)
    assert np.allclose(lmm.within_ratio(Y, d["Subject"]), e["within_ratio"], rtol=1e-9)
    # the decision is the moment ratio's, not the ML fit's: every column is still fitted by ML
    # where kept, and the fit drops exactly the flagged ones
    fit = model.fit(Y)
    assert np.array_equal(np.isnan(fit.stat), info["degenerate"])
    assert np.array_equal(fit.degenerate, info["degenerate"])
