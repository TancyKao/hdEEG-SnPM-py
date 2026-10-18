"""Partial correlation (correlationP/S with covariates) under the Freedman-Lane scheme.

Restates the MATLAB suite ``test_corr_freedman_lane`` (Method Spec approved 2026-10-18):

- F1 observed map equals an independent partial correlation (least-squares residuals, numpy and
  scipy), Pearson and Spearman, a channel with a missing cell stays excluded; uncorrected p uses
  df n - 2 - k.
- F2 a duplicated covariate gives k_eff = rank - 1, the same map and p, and a warning.
- F3 regression-preset oracle: legacy max-|r| FWE p equals the GLM regression max-|t| FWE p on
  the same permutation orders.
- F4 guards: n - k - 2 < 1 raises; n - k - 2 < 5 warns and the note reaches the results.
- F5 k = 0 is untouched: no Freedman-Lane context, map and p as the zero-order engine.
- F6 text: Freedman-Lane / approximate, never "exact"; Spearman caveat for correlationS only.
- F7 the engines refuse a Freedman-Lane context for a t-test and for correlationS.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hdeeg_snpm import glm, legacy, pipeline
from hdeeg_snpm.output import run_and_write
from test_invariants import N_CH, params

N, K = 20, 3
IDS = [f"sub{i:03d}" for i in range(1, N + 1)]


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def write_cov(folder, Z, name="cov", ids=None):
    frame = pd.DataFrame(Z, columns=[f"c{i + 1}" for i in range(Z.shape[1])])
    frame.insert(0, "Subject", ids or [f"sub{i:03d}" for i in range(1, len(Z) + 1)])
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def data():
    """n = 20, 3 covariates that drive both x and y; channel 7 has one missing cell."""
    rng = np.random.default_rng(101)
    Z = rng.standard_normal((N, K))
    X = Z @ rng.standard_normal((K, N_CH)) * 2 + rng.standard_normal((N, N_CH))
    Y = Z @ rng.standard_normal((K, N_CH)) * 2 + rng.standard_normal((N, N_CH)) + 0.4 * X
    X[4, 7] = np.nan
    return X, Y, Z


def partial_reference(X, Y, Z, spearman):
    """Per-channel partial correlation and its p, written independently of the package:
    least-squares residuals on [1, Z], scipy Pearson r, t with df n - 2 - k."""
    if spearman:
        X, Y, Z = (stats.rankdata(a, axis=0) for a in (X, Y, Z))
    n, k = Z.shape
    D = np.column_stack([np.ones(n), Z])
    r = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        if not (np.isfinite(X[:, c]).all() and np.isfinite(Y[:, c]).all()):
            continue
        ex = X[:, c] - D @ np.linalg.lstsq(D, X[:, c], rcond=None)[0]
        ey = Y[:, c] - D @ np.linalg.lstsq(D, Y[:, c], rcond=None)[0]
        r[c] = stats.pearsonr(ex, ey)[0]
    df = n - 2 - k
    with np.errstate(invalid="ignore"):
        p = 2 * stats.t.sf(np.abs(r * np.sqrt(df / (1 - r ** 2))), df)
    return r, p


def cov_params(tmp_path, comparison, X, Y, Z, **extra):
    return params(tmp_path, comparison, X, Y, use_covariates=True,
                  covariate_file=write_cov(tmp_path, Z), permutations=50, seed=3, **extra)


@pytest.mark.parametrize("comparison", ["correlationP", "correlationS"])
def test_f1_observed_map_and_p_equal_independent_partial_correlation(tmp_path, data, comparison):
    X, Y, Z = data
    res = quiet(pipeline.run_legacy, cov_params(tmp_path, comparison, X, Y, Z))
    prep = res.prepared
    ref_r, ref_p = partial_reference(X, Y, Z, comparison == "correlationS")
    assert 7 in prep.rule.excluded.tolist() and np.isnan(res.T.real_T[7])
    ok = np.isfinite(ref_r)
    assert np.array_equal(np.isfinite(res.T.real_T), ok)
    np.testing.assert_allclose(res.T.real_T[ok], ref_r[ok], rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.T.p_real[ok], ref_p[ok], rtol=1e-10, atol=1e-14)
    assert prep.stat_comparison == "correlationP" and prep.fl is not None
    assert (prep.ncov, prep.covariates.k_eff, prep.covariates.error_df) == (K, K, N - 2 - K)


def test_observed_map_is_the_identity_permutation(tmp_path, data):
    """The observed map goes through the same expression as the permuted maps, so the identity
    order reproduces it bit for bit."""
    X, Y, Z = data
    prep = quiet(pipeline.prepare, cov_params(tmp_path, "correlationP", X, Y, Z))
    design = legacy._design(prep.x, prep.y, "correlationP", "both", prep.fl)
    r_obs, _ = design.observed(True)
    r_id = design.stat(np.arange(N))
    assert np.array_equal(r_obs, r_id, equal_nan=True)


def test_f2_duplicated_covariate_counts_once(tmp_path, data):
    X, Y, Z = data
    base = quiet(pipeline.run_legacy, cov_params(tmp_path, "correlationP", X, Y, Z))
    p = params(tmp_path, "correlationP", X, Y, use_covariates=True,
               covariate_file=write_cov(tmp_path, np.column_stack([Z, Z[:, 0]]), "dup"),
               permutations=50, seed=3)
    with pytest.warns(UserWarning, match="span only 3 dimension"):
        dup = pipeline.run_legacy(p)
    assert dup.prepared.covariates.k_eff == K and dup.prepared.covariates.n_covariates == K + 1
    assert dup.prepared.covariates.error_df == N - 2 - K
    np.testing.assert_allclose(dup.T.real_T, base.T.real_T, rtol=0, atol=1e-12)
    np.testing.assert_allclose(dup.T.p_real, base.T.p_real, rtol=0, atol=1e-10)


def test_f3_regression_preset_oracle(tmp_path, data):
    """Single score: max-|r| FWE p of the legacy path equals the max-|t| FWE p of the GLM
    regression design (Freedman-Lane) on the same orders; the uncorrected p equals the
    regression t-test p."""
    X, _, Z = data
    rng = np.random.default_rng(7)
    Xc = X.copy()
    Xc[4, 7] = 0.3
    s = 1.5 * Xc[:, 30] + Z @ np.array([1.0, -1.0, 0.5]) + 0.5 * rng.standard_normal(N)
    score = tmp_path / "score.csv"
    pd.DataFrame({"Subject": IDS, "score": s}).to_csv(score, index=False)
    p = {**params(tmp_path, "correlationP", Xc, Xc), "data2_file": str(score),
         "use_covariates": True, "covariate_file": write_cov(tmp_path, Z)}
    orders = legacy._perm.row_permutations(N, 300, np.random.default_rng(2026))
    res = quiet(pipeline.run_legacy, p, perms_tfce=orders, perms_cluster=orders)
    assert res.prepared.rule.excluded.size == 0

    meta = pd.DataFrame({"score": s, "c1": Z[:, 0], "c2": Z[:, 1], "c3": Z[:, 2]})
    d = glm.build_design("regression", meta, predictor_col="score",
                         covariate_cols=["c1", "c2", "c3"])
    Y = Xc                                  # columns are the analysed channels, in order
    ctx = glm.fl_context(Y, d)
    tmax = np.array([np.max(np.abs(glm.glm_stat(ctx.fit + ctx.resid[o], d.X, d.C)[0]))
                     for o in orders])
    tobs, pobs, _, _ = glm.glm_stat(Y, d.X, d.C)
    tmax = np.sort(tmax)[::-1]
    pg = legacy._fwe_p(tobs, tmax, True)
    assert np.array_equal(pg, res.T.p_corrected)
    np.testing.assert_allclose(res.T.p_real, pobs, rtol=1e-10, atol=1e-14)


def test_f4_no_error_df_is_refused(tmp_path):
    rng = np.random.default_rng(4)
    x, y, Z = (rng.standard_normal((6, N_CH)), rng.standard_normal((6, N_CH)),
               rng.standard_normal((6, 4)))
    with pytest.raises(pipeline.CovariateNoErrorDF, match="leaves 0 error degrees"):
        quiet(pipeline.prepare, cov_params(tmp_path, "correlationP", x, y, Z))


def test_f4_low_df_warns_and_reaches_the_outputs(tmp_path):
    rng = np.random.default_rng(5)
    x, y, Z = (rng.standard_normal((8, N_CH)), rng.standard_normal((8, N_CH)),
               rng.standard_normal((8, 3)))
    p = {**cov_params(tmp_path, "correlationP", x, y, Z), "output_path": str(tmp_path / "out")}
    with pytest.warns(UserWarning, match=r"Only 3 error degrees of freedom .* APPROXIMATE"):
        prep = pipeline.prepare(p)
    assert prep.covariates.conservative_note == pipeline._conservative_note(3, 3, 8)
    paths = quiet(run_and_write, p)
    summary = json.loads(paths["json"].read_text())
    assert summary["effects"][0]["covariates"]["conservative_note"].startswith("Only 3 error")
    assert any("APPROXIMATE (Freedman-Lane)" in n for n in summary["notes"])
    assert "APPROXIMATE (Freedman-Lane)" in paths["html"].read_text(encoding="utf-8")


def test_f5_no_covariates_is_untouched(tmp_path, data):
    X, Y, _ = data
    res = quiet(pipeline.run_legacy, {**params(tmp_path, "correlationP", X, Y),
                                      "permutations": 50, "seed": 3})
    prep = res.prepared
    assert prep.fl is None and prep.covariates is None and prep.ncov == 0
    r0 = legacy.corr_columns(prep.x, prep.y)
    assert np.array_equal(res.T.real_T, r0, equal_nan=True)
    ok = np.isfinite(r0)
    p0 = np.full(r0.shape, np.nan)
    p0[ok] = [stats.pearsonr(prep.x[:, c], prep.y[:, c])[1] for c in np.flatnonzero(ok)]
    np.testing.assert_allclose(res.T.p_real, p0, rtol=1e-10)
    # the engines without a context are the zero-order engines, draw for draw
    a = legacy.single_threshold_tfce(prep.x, prep.y, prep.setup.graph, "correlationP",
                                     perms=res.T.perms)
    b = legacy.single_threshold_tfce(prep.x, prep.y, prep.setup.graph, "correlationP",
                                     perms=res.T.perms, fl=None)
    assert np.array_equal(a.tMax, b.tMax) and np.array_equal(a.tMax, res.T.tMax)


@pytest.mark.parametrize("comparison", ["correlationP", "correlationS"])
def test_f6_text_says_approximate_never_exact(tmp_path, data, comparison):
    X, Y, Z = data
    p = {**cov_params(tmp_path, comparison, X, Y, Z), "output_path": str(tmp_path / comparison)}
    paths = quiet(run_and_write, p)
    html = paths["html"].read_text(encoding="utf-8")
    cov = json.loads(paths["json"].read_text())["effects"][0]["covariates"]
    assert "Freedman-Lane, approximate" in cov["method"] and "Freedman-Lane, approximate" in html
    assert "exact test" not in html.replace("Not an exact test", "")
    has_caveat = "Partial Spearman controls the covariates" in html
    assert has_caveat == (comparison == "correlationS")
    assert bool(cov["spearman_caveat"]) == (comparison == "correlationS")


@pytest.mark.parametrize("comparison", ["pairedT", "unpairedT", "onesampleT", "correlationS"])
def test_f7_engines_refuse_a_context_off_correlationP(egi_graph, comparison):
    a = np.random.default_rng(0).standard_normal((10, egi_graph.n))
    fl = legacy.FLContext(np.eye(10), 0)
    with pytest.raises(ValueError, match="Freedman-Lane"):
        legacy.single_threshold_tfce(a, a, egi_graph, comparison, n_perm=10, fl=fl)
    with pytest.raises(ValueError, match="Freedman-Lane"):
        legacy.cluster_analysis(a, a, 0.5, egi_graph, comparison, n_perm=10, fl=fl)


def test_zero_variance_rule_inside_the_loop():
    """A column whose reordered residual falls inside the covariate space has nothing left
    after re-projection: it scores 0 for that order, not NaN or a rounding-noise r."""
    n = 8
    z = np.zeros(n)
    z[:2] = [1.0, -1.0]
    Rz, k = pipeline._residual_forming(z[:, None])
    x = np.zeros((n, 2))
    x[2:4, 0] = [1.0, -1.0]                 # orthogonal to [1, z], so Rz x = x
    x[:, 1] = np.random.default_rng(1).standard_normal(n)
    y = np.random.default_rng(2).standard_normal((n, 2))
    xt, yt = pipeline._residualise(x, Rz), pipeline._residualise(y, Rz)
    design = legacy._Correlation(xt, yt, spearman=False, fl=legacy.FLContext(Rz, k))
    order = np.array([2, 3, 0, 1, 4, 5, 6, 7])  # moves column 0 onto z itself
    r = design.stat(order)
    assert r[0] == 0.0 and np.isfinite(r[1]) and r[1] != 0.0
    assert np.isfinite(design.stat(np.arange(n))[0])


def test_residual_forming_matrix(data):
    _, _, Z = data
    Rz, k = pipeline._residual_forming(Z)
    D = np.column_stack([np.ones(N), Z])
    assert k == K
    np.testing.assert_allclose(Rz @ D, 0, atol=1e-12)
    np.testing.assert_allclose(Rz @ Rz, Rz, atol=1e-12)
    np.testing.assert_allclose(Rz, Rz.T, atol=1e-14)
    _, k2 = pipeline._residual_forming(np.column_stack([Z, Z[:, 0] + Z[:, 1], np.ones(N)]))
    assert k2 == K
