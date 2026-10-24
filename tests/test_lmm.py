"""Mixed linear model tier: the fit against independent fits, the boundary, planted effects under
the right and the wrong permutation scheme, and the fixed channel set."""

import warnings

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import lmm
from hdeeg_snpm.lmm import LMM, LMMSpec


def cluster_of(graph, centre=23):
    return sorted({centre, *graph.neighbor_lists[centre]})


def long_table(n_subj, n_trials, n_ch, rng, sd_subject=1.0, slope=0.0, group_shift=0.0,
               n_groups=2, cluster=None, latent_gain=1.2):
    """Trial-level metadata plus a (rows, channels) power matrix. With ``cluster`` the channels in
    it share a latent trial value that also drives the response (``slope``)."""
    subj = np.repeat(np.arange(n_subj), n_trials)
    grp = np.array([f"G{s % n_groups + 1}" for s in range(n_subj)])[subj]
    L = rng.standard_normal(subj.size)
    power = 10 + 2 * rng.standard_normal((subj.size, n_ch))
    if cluster is not None:
        power[:, cluster] += latent_gain * L[:, None]
    shift = np.array([group_shift * (int(g[1:]) - 1) for g in grp])
    dv = slope * L + shift + sd_subject * rng.standard_normal(n_subj)[subj] \
        + 0.5 * rng.standard_normal(subj.size)
    meta = pd.DataFrame({"Subject": [f"sub{s:02d}" for s in subj], "group": grp, "DV": dv})
    return meta, power


def dense_gls(y, X, groups, tau2, sigma2):
    """Coefficients, conditional covariance and log-likelihood from the full V matrix."""
    Z = (groups[:, None] == np.unique(groups)[None, :]).astype(float)
    V = sigma2 * np.eye(y.size) + tau2 * Z @ Z.T
    Vi = np.linalg.inv(V)
    cov = np.linalg.inv(X.T @ Vi @ X)
    beta = cov @ X.T @ Vi @ y
    r = y - X @ beta
    loglik = -0.5 * (y.size * np.log(2 * np.pi) + np.linalg.slogdet(V)[1] + r @ Vi @ r)
    return beta, cov, loglik


# --------------------------------------------------------------------------- the fit

def test_fit_matches_an_independent_mixed_model_fit():
    MixedLM = pytest.importorskip("statsmodels.regression.mixed_linear_model").MixedLM
    rng = np.random.default_rng(11)
    meta, power = long_table(15, 12, 6, rng, sd_subject=1.0)
    meta["DV"] += 0.3 * power[:, 0]
    model = LMM(meta, LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous"))
    fit = model.fit(power)
    groups = model.gidx
    for j in range(power.shape[1]):
        X = np.column_stack([np.ones(len(meta)), power[:, j]])
        y = meta["DV"].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = MixedLM(y, X, groups=groups).fit(reml=False, gtol=1e-10)
        tau2 = float(np.asarray(ref.cov_re).ravel()[0])
        assert fit.beta[j] == pytest.approx(np.asarray(ref.fe_params), rel=1e-6, abs=1e-9)
        assert fit.sigma[j] ** 2 == pytest.approx(ref.scale, rel=1e-6)
        assert fit.sd_subject[j] ** 2 == pytest.approx(tau2, rel=1e-5)
        assert fit.loglik[j] == pytest.approx(ref.llf, abs=1e-8)
        # standard error as fitlme defines it, from the reference's own variance components
        _, cov, loglik = dense_gls(y, X, groups, tau2, ref.scale)
        assert fit.se[j] == pytest.approx(np.sqrt(np.diag(cov)), rel=1e-5)
        assert fit.stat[j] == pytest.approx(fit.beta[j, 1] / np.sqrt(cov[1, 1]), rel=1e-5)
        assert loglik == pytest.approx(ref.llf, abs=1e-8)
    assert fit.df == len(meta) - 2


def test_factor_f_is_the_marginal_wald_test_with_residual_df():
    rng = np.random.default_rng(3)
    meta, power = long_table(18, 6, 5, rng, n_groups=3, group_shift=0.8)
    model = LMM(meta, LMMSpec(dv="DV", fixed="POWER + group", effect="group",
                              effect_type="factor"))
    assert model.names == ["(Intercept)", "group_G2", "group_G3", "POWER"]
    fit = model.fit(power)
    y = meta["DV"].to_numpy()
    for j in range(power.shape[1]):
        X, _ = model.design(power[:, [j]])
        tau2 = fit.sd_subject[j] ** 2
        beta, cov, loglik = dense_gls(y, X[0], model.gidx, tau2, fit.sigma[j] ** 2)
        b = beta[1:3]
        F = b @ np.linalg.solve(cov[1:3, 1:3], b) / 2
        assert fit.stat[j] == pytest.approx(F, rel=1e-8)
        assert fit.wald[j] == fit.stat[j]
        assert fit.loglik[j] == pytest.approx(loglik, abs=1e-8)
    assert fit.df == len(meta) - 4
    from scipy import stats
    assert fit.p_model == pytest.approx(stats.f.sf(fit.stat, 2, len(meta) - 4))
    # one-df factor: F equals the square of the coefficient's t
    two = LMM(meta.assign(group=np.where(meta["group"] == "G1", "G1", "G2")),
              LMMSpec(dv="DV", fixed="POWER + group", effect="group", effect_type="factor"))
    cont = LMM(two.meta, LMMSpec(dv="DV", fixed="POWER + group", effect="group",
                                 effect_type="continuous"))
    assert two.fit(power).stat == pytest.approx(cont.fit(power).stat ** 2, rel=1e-10)


def test_zero_subject_variance_is_found_exactly_and_equals_least_squares():
    rng = np.random.default_rng(5)
    meta, power = long_table(10, 10, 40, rng, sd_subject=0.0)
    e = rng.standard_normal(len(meta))
    e -= pd.Series(e).groupby(meta["Subject"]).transform("mean").to_numpy()  # no subject spread
    meta["DV"] = 0.2 * power[:, 0] + e
    fit = LMM(meta, LMMSpec(dv="DV", fixed="POWER", effect="POWER",
                            effect_type="continuous")).fit(power)
    assert fit.boundary.sum() >= 10
    y, n = meta["DV"].to_numpy(), len(meta)
    for j in np.flatnonzero(fit.boundary):
        assert fit.theta[j] == 0.0 and fit.sd_subject[j] == 0.0
        X = np.column_stack([np.ones(n), power[:, j]])
        beta, rss = np.linalg.lstsq(X, y, rcond=None)[:2]
        cov = rss[0] / n * np.linalg.inv(X.T @ X)        # ML variance, as fitlme
        assert fit.stat[j] == pytest.approx(beta[1] / np.sqrt(cov[1, 1]), rel=1e-10)
    # a channel with real subject spread stays off the boundary
    meta["DV"] += 2.0 * rng.standard_normal(10)[LMM(meta, LMMSpec(
        dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")).gidx]
    fit = LMM(meta, LMMSpec(dv="DV", fixed="POWER", effect="POWER",
                            effect_type="continuous")).fit(power)
    assert not fit.boundary.any() and (fit.theta > 0.5).all()


def test_unfit_channels_are_nan_with_zero_wald():
    rng = np.random.default_rng(2)
    meta, power = long_table(8, 6, 4, rng)
    power[:, 1] = 3.0                                     # rank-deficient design
    fit = LMM(meta, LMMSpec(dv="DV", fixed="POWER", effect="POWER",
                            effect_type="continuous")).fit(power)
    assert np.isnan(fit.stat[1]) and fit.wald[1] == 0 and np.isnan(fit.p_model[1])
    assert np.isfinite(fit.stat[[0, 2, 3]]).all()
    few = LMM(meta.iloc[:4], LMMSpec(dv="DV", fixed="POWER", effect="POWER",
                                     effect_type="continuous")).fit(power[:4])
    assert np.isnan(few.stat).all() and (few.wald == 0).all()   # fewer than 5 rows


@pytest.mark.parametrize("random", ["(1|Subject) + (1|time)", "(POWER|Subject)",
                                    "(1 + POWER|Subject)", "", "(1|Subject:time)"])
def test_other_random_structures_are_refused(random):
    with pytest.raises(lmm.UnsupportedRandomEffects, match="random intercept"):
        LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous",
                random=random)


def test_specification_errors_name_the_problem():
    rng = np.random.default_rng(0)
    meta, _ = long_table(6, 5, 3, rng)
    with pytest.raises(ValueError, match="not a term"):
        LMM(meta, LMMSpec(dv="DV", fixed="POWER", effect="group", effect_type="factor"))
    bad = meta.copy()
    bad.loc[3, "group"] = None
    with pytest.raises(ValueError, match="'group' has a missing value in row"):
        LMM(bad, LMMSpec(dv="DV", fixed="POWER + group", effect="POWER",
                         effect_type="continuous"))
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="factor")
    assert spec.perm == "group_label"
    assert LMMSpec(dv="DV", fixed="POWER", effect="POWER",
                   effect_type="continuous").perm == "within_subject"


def test_interaction_columns_and_continuous_effect_lookup():
    rng = np.random.default_rng(1)
    meta, power = long_table(8, 5, 2, rng)
    model = LMM(meta, LMMSpec(dv="DV", fixed="POWER*group", effect="POWER",
                              effect_type="continuous"))
    assert model.names == ["(Intercept)", "group_G2", "POWER", "group_G2:POWER"]
    assert model.effect_cols == [2]
    X, _ = model.design(power)
    g2 = (meta["group"] == "G2").to_numpy()
    assert np.array_equal(X[1, :, 3], power[:, 1] * g2)


# --------------------------------------------------------------------------- permutations

def test_permutation_schemes_follow_snpm_lmm_permute_meta():
    rng = np.random.default_rng(4)
    meta, _ = long_table(6, 4, 1, rng)
    spec = LMMSpec(dv="DV", fixed="POWER + group", effect="POWER", effect_type="continuous")
    perms = lmm.draw_perms(meta, spec, 50, rng)
    subj = meta["Subject"].to_numpy()
    assert perms.shape == (50, len(meta))
    assert all((subj[row] == subj).all() for row in perms)          # only within a subject
    out = lmm.permute_meta(meta, spec, perms[1])
    assert np.array_equal(out["DV"].to_numpy(), meta["DV"].to_numpy()[perms[1]])

    gspec = LMMSpec(dv="DV", fixed="POWER + group", effect="group", effect_type="factor")
    gperms = lmm.draw_perms(meta, gspec, 50, rng)
    assert gperms.shape == (50, 6)
    out = lmm.permute_meta(meta, gspec, gperms[3])
    per_subject = out.groupby("Subject")["group"].nunique()
    assert (per_subject == 1).all()                                  # whole subjects relabelled
    assert sorted(out.drop_duplicates("Subject")["group"]) == sorted(
        meta.drop_duplicates("Subject")["group"])
    order = pd.unique(meta["Subject"])
    first = meta.drop_duplicates("Subject").set_index("Subject")["group"]
    got = out.drop_duplicates("Subject").set_index("Subject")["group"]
    assert [got[s] for s in order] == [first[order[k]] for k in gperms[3]]


def test_given_perms_reproduce_a_drawn_run(egi_graph):
    rng = np.random.default_rng(9)
    meta, power = long_table(6, 6, egi_graph.n, rng)
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    a = lmm.lmm_tfce(power, meta, spec, egi_graph, n_perm=8, rng=np.random.default_rng(1))
    b = lmm.lmm_tfce(power, meta, spec, egi_graph, perms=a.perms)
    assert np.array_equal(a.tMaxTFCE, b.tMaxTFCE)
    assert np.array_equal(a.p_correctedTFCE, b.p_correctedTFCE, equal_nan=True)
    c = lmm.lmm_cluster(power, meta, spec, egi_graph, n_perm=8, rng=np.random.default_rng(1))
    d = lmm.lmm_cluster(power, meta, spec, egi_graph, perms=c.perms)
    assert np.array_equal(c.maxMass, d.maxMass)


# --------------------------------------------------------------------------- inference

def test_planted_continuous_cluster_is_recovered(egi_graph):
    rng = np.random.default_rng(20)
    cl = cluster_of(egi_graph)
    meta, power = long_table(12, 10, egi_graph.n, rng, slope=1.0, cluster=cl)
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    t = lmm.lmm_tfce(power, meta, spec, egi_graph, n_perm=99, rng=rng)
    sig = set(np.flatnonzero(t.p_correctedTFCE < 0.05).tolist())
    assert len(set(cl) - sig) <= 1 and len(sig - set(cl)) <= 4
    c = lmm.lmm_cluster(power, meta, spec, egi_graph, n_perm=99, rng=rng)
    best = min(c.clusters, key=lambda k: k.p)
    assert best.p < 0.05 and len(set(cl) - set(best.channels.tolist())) <= 1
    assert best.mass == pytest.approx(t.fit.wald[best.channels].sum() / best.channels.size)


def test_group_effect_needs_the_group_label_scheme(egi_graph):
    """A between-group effect is found when whole subjects are relabelled; a within-subject
    shuffle leaves the group means in place, so it gives no null for it and finds nothing; and
    a null design finds nothing under the right scheme."""
    rng = np.random.default_rng(21)
    meta, power = long_table(16, 6, egi_graph.n, rng, sd_subject=0.3, group_shift=2.0)
    right = LMMSpec(dv="DV", fixed="POWER + group", effect="group", effect_type="factor")
    wrong = LMMSpec(dv="DV", fixed="POWER + group", effect="group", effect_type="factor",
                    perm="within_subject")
    hit = lmm.lmm_tfce(power, meta, right, egi_graph, n_perm=39, rng=rng)
    assert (hit.p_correctedTFCE < 0.05).sum() > 0.9 * egi_graph.n
    miss = lmm.lmm_tfce(power, meta, wrong, egi_graph, n_perm=39, rng=rng)
    assert (miss.p_correctedTFCE < 0.05).sum() == 0
    null_meta = meta.assign(DV=rng.permutation(meta["DV"].to_numpy()))
    null = lmm.lmm_tfce(power, null_meta, right, egi_graph, n_perm=39, rng=rng)
    assert (null.p_correctedTFCE < 0.05).sum() == 0
    null_c = lmm.lmm_cluster(power, null_meta, right, egi_graph, n_perm=39, rng=rng)
    assert all(k.p > 0.05 for k in null_c.clusters)


# --------------------------------------------------------------------------- channel set

def test_incomplete_channels_are_out_of_every_map(egi_graph):
    """As test_lmm_invariant.m: the evaluable set is the complete columns, fixed before the loop;
    what an excluded channel holds cannot change any number."""
    rng = np.random.default_rng(30)
    meta, power = long_table(8, 6, egi_graph.n, rng, slope=0.5, cluster=cluster_of(egi_graph))
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    holes = power.copy()
    holes[3, [5, 40]] = np.nan
    holes[:, 77] = np.nan
    t = lmm.lmm_tfce(holes, meta, spec, egi_graph, n_perm=20, rng=np.random.default_rng(0))
    assert list(t.excluded_channels) == [5, 40, 77]
    assert t.excluded_reasons == [lmm.MISSING] * 3
    assert np.isnan(t.real_T[[5, 40, 77]]).all() and np.isnan(t.p_correctedTFCE[[5, 40, 77]]).all()
    filled = power.copy()
    filled[:, [5, 40, 77]] = rng.standard_normal((len(meta), 3)) * 50
    ev = np.isfinite(holes).all(axis=0)
    u = lmm.lmm_tfce(filled, meta, spec, egi_graph, perms=t.perms, evaluable=ev)
    assert np.array_equal(t.tMaxTFCE, u.tMaxTFCE)
    assert np.array_equal(t.real_T, u.real_T, equal_nan=True)
    c1 = lmm.lmm_cluster(holes, meta, spec, egi_graph, n_perm=20, rng=np.random.default_rng(1))
    c2 = lmm.lmm_cluster(filled, meta, spec, egi_graph, perms=c1.perms, evaluable=ev)
    assert np.array_equal(c1.maxMass, c2.maxMass)
    # on complete data the explicit all-true mask changes nothing
    a = lmm.lmm_tfce(power, meta, spec, egi_graph, perms=t.perms)
    b = lmm.lmm_tfce(power, meta, spec, egi_graph, perms=t.perms,
                     evaluable=np.ones(egi_graph.n, bool))
    assert np.array_equal(a.tMaxTFCE, b.tMaxTFCE)
    nothing = np.full_like(power, np.nan)
    with pytest.raises(ValueError, match="No channel is complete"):
        lmm.lmm_tfce(nothing, meta, spec, egi_graph, n_perm=5)


def test_a_permutation_without_any_statistic_enters_the_null_as_inf(egi_graph):
    rng = np.random.default_rng(31)
    meta, power = long_table(6, 5, egi_graph.n, rng)
    power[:] = 2.0                                         # every design rank deficient
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    with pytest.warns(RuntimeWarning, match="no finite statistic"):
        t = lmm.lmm_tfce(power, meta, spec, egi_graph, n_perm=5, rng=rng)
    assert np.isinf(t.tMaxTFCE).all() and t.n_degenerate == 5
    assert np.isnan(t.p_correctedTFCE).all()
    assert lmm._null_max(np.array([np.nan, 1.0, -3.0]), np.array([True, True, False]),
                         "both") == 1.0
    assert lmm._null_max(np.array([np.nan, -2.0]), np.array([True, True]), "both") == 2.0
    assert lmm._null_max(np.array([np.nan, -2.0]), np.array([True, True]), "right") == -2.0


# --------------------------------------------------------------------------- missing metadata

def _drop(meta, power, **kw):
    spec = LMMSpec(dv="DV", fixed=kw.pop("fixed", "POWER + group"), effect=kw.pop("effect", "POWER"),
                   effect_type=kw.pop("effect_type", "continuous"), **kw)
    return lmm.drop_missing_meta(meta, spec, power)


def test_rows_with_a_missing_model_variable_are_dropped_up_front():
    rng = np.random.default_rng(40)
    meta, power = long_table(6, 5, 4, rng)
    meta["time"] = np.tile(np.arange(5), 6).astype(float)
    meta.loc[2, "DV"] = np.nan
    meta.loc[7, "group"] = ""                  # empty string counts as missing
    meta.loc[[25, 26, 27, 28, 29], "group"] = None   # every row of the last subject
    meta.loc[11, "time"] = np.nan              # not a model variable: kept
    power[3, 1] = np.nan                       # channel data: left to the evaluable mask
    with pytest.warns(lmm.LMMMetaNaNRows) as w:
        out, pw, info = _drop(meta, power)
    assert str(w[0].message) == (
        "7 of 30 rows (trials) dropped for a missing model variable (group 6, DV 1); subjects "
        "that lost every trial: sub05. The dropped rows are left out of the observed fit and "
        "every permutation alike; this assumes the missingness is unrelated to the effect "
        "tested.")
    assert list(info.rows) == [2, 7, 25, 26, 27, 28, 29]
    assert info.n == 7 and info.n_input == 30 and len(out) == pw.shape[0] == 23
    assert info.n_subjects == 5 and info.subjects_lost == ["sub05"]
    assert np.array_equal(pw, np.delete(power, info.rows, axis=0), equal_nan=True)
    assert info.columns == ["Subject", "group", "DV"] and info.per_column == [0, 6, 1]
    assert lmm.dropped_text(info) == ("Rows dropped for a missing model variable: 7 of 30 "
                                      "(group 6, DV 1); subjects lost entirely: sub05")
    cols = list(meta.columns)
    # a response of POWER is the channel; identifiers that are not metadata columns are ignored
    spec = LMMSpec(dv="POWER", fixed="group", effect="group", effect_type="factor")
    assert lmm.model_variables(spec, cols) == ["Subject", "group"]
    gspec = LMMSpec(dv="DV", fixed="POWER + age", effect="POWER", effect_type="continuous",
                    perm="group_label")
    assert lmm.model_variables(gspec, cols) == ["Subject", "group", "DV"]


def test_nothing_missing_drops_nothing():
    rng = np.random.default_rng(44)
    meta, power = long_table(5, 4, 2, rng)
    out, pw, info = _drop(meta, power)
    assert info.n == 0 and out.equals(meta) and np.array_equal(pw, power)
    assert lmm.dropped_text(info) == "Rows dropped for a missing model variable: none"


def test_design_checks_after_the_drop():
    rng = np.random.default_rng(41)
    meta, power = long_table(6, 4, 3, rng)
    few = meta.copy()
    few.loc[few["Subject"].isin(["sub00", "sub01", "sub02", "sub03"]), "DV"] = np.nan
    with pytest.raises(lmm.LMMInputError, match="Only 2 subject") as e, \
            pytest.warns(lmm.LMMMetaNaNRows):
        _drop(few, power)
    assert e.value.code == "tooFewSubjects"
    varies = meta.copy()
    varies.loc[1, "group"] = "G2"              # sub00 is G1 on its other rows; nothing missing
    with pytest.raises(lmm.LMMInputError, match="it varies within 1 subject") as e:
        _drop(varies, power, effect="group", effect_type="factor")
    assert e.value.code == "groupVariesWithinSubject"
    _drop(varies, power)                       # within_subject does not relabel groups
    lonely = meta.assign(group=np.where(meta["Subject"] == "sub00", "G3", meta["group"]))
    with pytest.raises(lmm.LMMInputError, match="per group after the drop: G1 2, G2 3, G3 1") as e:
        _drop(lonely, power, effect="group", effect_type="factor")
    assert e.value.code == "groupLevelTooFewSubjects"
    one = meta.assign(group="G1")
    with pytest.raises(lmm.LMMInputError, match="at least 2 groups"):
        _drop(one, power, effect="POWER", perm="group_label")
    with pytest.raises(lmm.LMMInputError) as e:
        _drop(meta.drop(columns="Subject"), power)
    assert e.value.code == "noSubjectColumn"
    with pytest.raises(lmm.LMMInputError) as e:
        _drop(meta.drop(columns="group"), power, fixed="POWER", perm="group_label")
    assert e.value.code == "noGroupColumn"
    single = meta.drop(index=[1, 2, 3]).reset_index(drop=True)
    with pytest.warns(lmm.LMMSingleTrialSubject, match=r"^1 subject\(s\) have a single trial "
                      r"\(sub00\): the within-subject shuffle cannot move their DV"):
        _drop(single, power[:single.shape[0]])


def test_identity_permutation_returns_the_metadata_unchanged():
    rng = np.random.default_rng(42)
    meta, _ = long_table(5, 4, 1, rng)
    for scheme, n in [("within_subject", len(meta)), ("group_label", 5)]:
        spec = LMMSpec(dv="DV", fixed="POWER + group", effect="POWER", effect_type="continuous",
                       perm=scheme)
        assert lmm.permute_meta(meta, spec, np.arange(n)) is meta


# --------------------------------------------------------------------------- pipeline

def test_mixedmodel_runs_through_the_pipeline_and_writes_its_files(egi_graph, tmp_path):
    import json

    from hdeeg_snpm.montage import get_montage
    from hdeeg_snpm.output import run_and_write

    rng = np.random.default_rng(43)
    cl = cluster_of(egi_graph)
    meta, power = long_table(10, 8, egi_graph.n, rng, slope=1.0, cluster=cl)
    meta.loc[[0, 1], "DV"] = np.nan
    power[5, 60] = np.nan
    table = pd.concat([meta, pd.DataFrame(power, columns=get_montage("egi").labels)], axis=1)
    table.to_csv(tmp_path / "long.csv", index=False)
    params = {"comparison": "mixedmodel", "data_file": str(tmp_path / "long.csv"),
              "channels": "egi", "lmm_meta_cols": ["Subject", "group", "DV"], "lmm_dv": "DV",
              "lmm_fixed": "POWER", "lmm_random": "(1|Subject)", "lmm_effect": "POWER",
              "lmm_effect_type": "continuous", "permutations": 39, "seed": 5,
              "output_path": str(tmp_path / "out")}
    with pytest.warns(lmm.LMMMetaNaNRows, match="2 of 80 rows"):
        paths = run_and_write(params)
    assert all(p.exists() for p in paths.values())
    summary = json.loads(paths["json"].read_text())
    e = summary["effects"][0]
    mm = e["mixed_model"]
    assert mm["perm_scheme"] == "within_subject" and mm["n_trials"] == 78
    assert mm["dropped_rows"]["rows"] == [0, 1] and mm["n_subjects"] == 10
    assert e["excluded_channels"] == [get_montage("egi").labels[60]]
    assert np.isfinite(e["global"]["statistic"]) and e["global"]["n_channels"] == egi_graph.n - 1
    assert set(e["significant_channels"]["tfce"]) & {get_montage("egi").labels[i] for i in cl}
    html_text = paths["html"].read_text()
    assert "mean Wald" in html_text and "within_subject" in html_text
    assert "random intercept per Subject" in html_text
    sheets = pd.read_excel(paths["xlsx"], sheet_name=None)
    assert {"statVal", "excludedChannels", "mixedModel", "droppedRows"} <= set(sheets)
    assert sheets["droppedRows"]["Value"].iloc[0] == (
        "Rows dropped for a missing model variable: 2 of 80 (DV 2)")
    assert "Rows dropped" in html_text and "2 of 80 (DV 2)" in html_text
    with pytest.raises(lmm.UnsupportedRandomEffects):
        from hdeeg_snpm import pipeline
        pipeline.run(dict(params, lmm_random="(1|Subject) + (1|time)"))


# --------------------------------------------------------------------------- degenerate channels

def _flat_case(egi_graph):
    """Factor effect on the channel value (group_label); channel 2 flat, channel 3 a per-subject
    constant, so both leave no residual variance."""
    rng = np.random.default_rng(50)
    n_subj, n_tr = 12, 6
    subj = np.repeat(np.arange(n_subj), n_tr)
    meta = pd.DataFrame({"Subject": [f"s{s:02d}" for s in subj],
                         "group": np.array(["A", "B", "C"] * 4)[subj]})
    power = rng.standard_normal((subj.size, egi_graph.n))
    power[:, 2] = 1.0
    power[:, 3] = rng.standard_normal(n_subj)[subj]
    spec = LMMSpec(dv="POWER", fixed="group", effect="group", effect_type="factor")
    return meta, power, spec


def test_a_channel_without_residual_variance_has_no_statistic(egi_graph):
    meta, power, spec = _flat_case(egi_graph)
    model = LMM(meta, spec)
    fit = model.fit(power)
    assert np.isnan(fit.stat[[2, 3]]).all() and (fit.wald[[2, 3]] == 0).all()
    assert np.isnan(fit.p_model[[2, 3]]).all()
    assert np.isfinite(fit.stat[[0, 1, 4]]).all()
    for row in lmm.draw_perms(meta, spec, 10, np.random.default_rng(1)):
        f = model.fit(power, lmm.permute_meta(meta, spec, row))
        assert np.isnan(f.stat[[2, 3]]).all() and (f.wald[[2, 3]] == 0).all()
        assert np.isfinite(f.stat[[0, 1, 4]]).all()
    cont = LMM(meta.assign(DV=np.random.default_rng(2).standard_normal(len(meta))),
               LMMSpec(dv="POWER", fixed="DV", effect="DV", effect_type="continuous",
                       perm="group_label"))
    assert np.isnan(cont.fit(power).stat[[2, 3]]).all()


def test_runs_complete_with_degenerate_channels(egi_graph):
    meta, power, spec = _flat_case(egi_graph)
    t = lmm.lmm_tfce(power, meta, spec, egi_graph, n_perm=20, rng=np.random.default_rng(3))
    assert np.isnan(t.real_T[[2, 3]]).all() and np.isnan(t.p_correctedTFCE[[2, 3]]).all()
    assert np.isfinite(t.tMaxTFCE).all()
    c = lmm.lmm_cluster(power, meta, spec, egi_graph, n_perm=20, rng=np.random.default_rng(4))
    assert all(2 not in k.channels and 3 not in k.channels for k in c.clusters)


# --------------------------------------------------------------------------- up-front refusals

def test_settings_the_permutation_test_cannot_handle_are_refused(egi_graph):
    rng = np.random.default_rng(51)
    meta, power = long_table(6, 5, egi_graph.n, rng)
    cont = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    for tail in ("left", "right"):
        with pytest.raises(lmm.LMMInputError) as e:
            lmm.lmm_tfce(power, meta, cont, egi_graph, tail=tail, n_perm=20)
        assert e.value.code == "oneSidedTail"
    wrong = LMMSpec(dv="DV", fixed="POWER + group", effect="POWER", effect_type="continuous",
                    perm="group_label")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.lmm_tfce(power, meta, wrong, egi_graph, n_perm=20)
    assert e.value.code == "effectNotPermuted"
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.lmm_cluster(power, meta, wrong, egi_graph, n_perm=20)
    assert e.value.code == "effectNotPermuted"
    lmm.check_design(LMMSpec(dv="DV", fixed="POWER*group", effect="POWER:group",
                             effect_type="factor"))          # contains the group column: fine
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(LMMSpec(dv="POWER", fixed="group", effect="group",
                                 effect_type="factor", perm="within_subject"))
    assert e.value.code == "dvNotMetadata"
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(cont, meta_columns=["Subject", "group"])
    assert e.value.code == "dvNotMetadata"


def test_pipeline_refuses_before_reading_any_fit(tmp_path):
    from hdeeg_snpm import pipeline

    base = {"comparison": "mixedmodel", "data_file": str(tmp_path / "absent.csv"),
            "lmm_meta_cols": ["Subject", "group", "DV"], "lmm_dv": "DV", "lmm_fixed": "POWER",
            "lmm_random": "(1|Subject)", "lmm_effect": "POWER",
            "lmm_effect_type": "continuous", "permutations": 20}
    for extra, code in [({"tail": "left"}, "oneSidedTail"),
                        ({"lmm_perm": "group_label"}, "effectNotPermuted"),
                        ({"lmm_dv": "POWER", "lmm_fixed": "group", "lmm_effect": "group",
                          "lmm_effect_type": "factor", "lmm_perm": "within_subject"},
                         "dvNotMetadata")]:
        with pytest.raises(lmm.LMMInputError) as e:   # raised before the (absent) file is read
            pipeline.run(dict(base, **extra))
        assert e.value.code == code


def test_refusal_texts_are_matlabs():
    """Word for word as core_snpm_lmm.m (core_snpm_lmm:oneSidedTail, effectNotPermuted,
    dvNotMetadata in its two forms); the tail check is case-sensitive, as strcmp."""
    cont = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(cont, "Both")
    assert str(e.value) == (
        "mixedmodel requires tail = 'both' (got 'Both'). The effect map is a two-sided t or an F, "
        "and clusters are formed on the two-sided model p; a one-sided TFCE null of signed maxima "
        "would not match it.")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(LMMSpec(dv="DV", fixed="POWER + group", effect="POWER",
                                 effect_type="continuous", perm="group_label"))
    assert str(e.value) == (
        "The group_label scheme relabels subjects between levels of 'group', but the effect "
        "'POWER' does not contain 'group', so every permuted map would equal the observed one. "
        "Use within_subject for an effect that varies within subjects, or test an effect of the "
        "group column.")
    tail = " Use group_label for a between-group effect on the channel value."
    head = ("The within_subject scheme shuffles the response within each subject, so the "
            "response must be a metadata column; ")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(LMMSpec(dv="POWER", fixed="group", effect="group", effect_type="factor",
                                 perm="within_subject"))
    assert str(e.value) == head + "'POWER' is not (it is the channel value)." + tail
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(cont, meta_columns=["Subject"])
    assert str(e.value) == head + "'DV' is not among lmm_meta_cols." + tail


# --------------------------------------------------------------------------- constant within subject

def test_within_ratio():
    subj = np.repeat(["a", "b", "c"], 4)
    v = np.column_stack([np.full(12, 3.0),                       # flat: 0
                         np.repeat([1.0, 2.0, 5.0], 4),          # constant within subject: 0
                         np.tile([0.0, 1.0, 0.0, 1.0], 3),       # all within: 1
                         np.r_[np.nan, np.ones(11)]])            # NaN column: NaN
    w = lmm.within_ratio(v, subj)
    assert w[0] == 0 and w[1] == pytest.approx(0, abs=1e-30) and w[2] == pytest.approx(1)
    assert np.isnan(w[3])
    x = np.random.default_rng(0).standard_normal(12)
    m = pd.Series(x).groupby(subj).transform("mean").to_numpy()
    assert lmm.within_ratio(x, subj)[0] == pytest.approx(((x - m) ** 2).sum()
                                                         / ((x - x.mean()) ** 2).sum())


def _write(tmp_path, meta, power, labels):
    table = pd.concat([meta, pd.DataFrame(power, columns=labels)], axis=1)
    table.to_csv(tmp_path / "long.csv", index=False)
    return str(tmp_path / "long.csv")


def _params(path, **kw):
    p = {"comparison": "mixedmodel", "data_file": path, "channels": "egi",
         "lmm_meta_cols": ["Subject", "group", "DV"], "lmm_dv": "DV", "lmm_fixed": "POWER",
         "lmm_random": "(1|Subject)", "lmm_effect": "POWER", "lmm_effect_type": "continuous",
         "permutations": 20, "seed": 1}
    p.update(kw)
    return p


def test_constant_within_subject_is_excluded_or_refused_up_front(egi_graph, tmp_path):
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    labels = get_montage("egi").labels
    rng = np.random.default_rng(60)
    meta, power = long_table(8, 5, egi_graph.n, rng)
    subj = meta["Subject"].to_numpy()
    power[:, 7] = pd.Series(power[:, 7]).groupby(subj).transform("mean").to_numpy()
    power[:, 9] = 4.0
    power[0, 11] = np.nan                                     # missing: keeps its own reason
    path = _write(tmp_path, meta, power, labels)
    # predictor constant within subject under within_subject
    out = lmm_pipeline.prepare(_params(path))
    ev, reasons = out[4], out[7]
    assert reasons == {7: lmm.CONST_PREDICTOR, 9: lmm.CONST_PREDICTOR, 11: lmm.MISSING}
    assert not ev[[7, 9, 11]].any() and ev.sum() == egi_graph.n - 3
    # channel value as the response (group_label): the response reason
    out = lmm_pipeline.prepare(_params(path, lmm_dv="POWER", lmm_fixed="group",
                                       lmm_effect="group", lmm_effect_type="factor"))
    assert out[7] == {7: lmm.CONST_RESPONSE, 9: lmm.CONST_RESPONSE, 11: lmm.MISSING}
    # POWER as predictor under group_label: a subject-constant channel is kept; the flat one
    # (one value on every trial) is excluded with its own reason
    out = lmm_pipeline.prepare(_params(path, lmm_fixed="POWER + group", lmm_effect="group",
                                       lmm_effect_type="factor"))
    assert out[7] == {9: lmm.FLAT, 11: lmm.MISSING}
    # a DV constant within every subject cannot be shuffled
    flat_dv = meta.assign(DV=pd.Series(meta["DV"]).groupby(subj).transform("mean"))
    path2 = str(tmp_path / "dv.csv")
    pd.concat([flat_dv, pd.DataFrame(power, columns=labels)], axis=1).to_csv(path2, index=False)
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(_params(path2))
    assert e.value.code == "dvConstantWithinSubject"
    w = lmm.within_ratio(flat_dv["DV"].to_numpy(), subj)[0]
    assert str(e.value) == (
        f"The within_subject scheme shuffles 'DV' within each subject, but it is constant within "
        f"each subject (within-subject share of its variation {w:.3g}, threshold 1e-06), so the "
        "within-subject shuffle has nothing to permute. Use group_label for a between-group "
        "effect, or analyse subject means.")
    # nothing left
    allflat = np.repeat(np.arange(8.0), 5)[:, None] + np.zeros((40, egi_graph.n))
    path3 = _write(tmp_path, meta, allflat, labels)
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(_params(path3))
    assert e.value.code == "noEvaluableChannels"
    assert str(e.value) == (f"No channel is left to test: {egi_graph.n} of {egi_graph.n} "
                            f"excluded ({lmm.CONST_PREDICTOR}).")


def test_observed_backstop_hits_are_listed(egi_graph):
    meta, power, spec = _flat_case(egi_graph)
    t = lmm.lmm_tfce(power, meta, spec, egi_graph, n_perm=20, rng=np.random.default_rng(3))
    assert list(t.undefined_channels) == [2, 3]
    assert np.isnan(t.real_T[[2, 3]]).all()


def test_backstop_ignores_a_saturated_random_design():
    """One trial per subject: [X Z] has rank N, so it fits any response exactly and RSS(y ~ [X Z])
    = 0 says nothing; as in snpm_lmm_degenerate.m the rule then applies only when rank < N."""
    rng = np.random.default_rng(70)
    n = 10
    gidx = np.arange(n)
    X = np.stack([np.column_stack([np.ones(n), rng.standard_normal(n)]) for _ in range(2)])
    y = np.column_stack([rng.standard_normal(n), np.full(n, 2.0)])   # ordinary; flat
    info = lmm.degenerate_info(y, X, gidx, n)
    assert (info["rank_xz"] == n).all() and info["rss_xz"][0] <= 1e-20 * (y[:, 0] ** 2).sum()
    assert list(info["degenerate"]) == [False, True]   # the flat one by RSS(y ~ X)
    assert np.isnan(info["theta2"]).all()              # df_w = 0: no moment ratio


def test_effect_not_permuted_accepts_coefficient_names():
    """As core_snpm_lmm.m: a token passes if it is the group column or starts with it plus '_'."""
    for effect in ("group_B", "POWER:group_B", "group", "POWER*group"):
        lmm.check_design(LMMSpec(dv="DV", fixed="POWER*group", effect=effect,
                                 effect_type="continuous", perm="group_label"))
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_design(LMMSpec(dv="DV", fixed="POWER + groupSize", effect="groupSize",
                                 effect_type="continuous", perm="group_label"))
    assert e.value.code == "effectNotPermuted"
    assert str(e.value) == (
        "The group_label scheme relabels subjects between levels of 'group', but the effect "
        "'groupSize' does not contain 'group', so every permuted map would equal the observed "
        "one. Use within_subject for an effect that varies within subjects, or test an effect of "
        "the group column.")


# --------------------------------------------------------------------------- MATLAB T22-T27

def test_inf_in_a_numeric_model_column_is_dropped_like_nan():
    """MATLAB test_lmm_invariant T22: fitlme refuses an Inf row, which left every channel NaN; an
    Inf in a model column is dropped like a NaN, an Inf in another column drops nothing."""
    rng = np.random.default_rng(221)
    meta, power = long_table(8, 9, 3, rng)
    meta["time"] = np.tile(np.arange(9.0), 8)
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    inf = meta.copy()
    inf.loc[[3, 29], "DV"] = [np.inf, -np.inf]
    inf.loc[8, "time"] = np.inf                       # not a model variable
    with pytest.warns(lmm.LMMMetaNaNRows):
        m2, p2, d = lmm.drop_missing_meta(inf, spec, power)
    assert d.n == 2 and d.rows.tolist() == [3, 29] and len(m2) == 70 and p2.shape[0] == 70
    assert d.columns == ["Subject", "DV"] and d.per_column == [0, 2]
    nan = meta.copy()
    nan.loc[[3, 29], "DV"] = np.nan
    with pytest.warns(lmm.LMMMetaNaNRows):
        m3, p3, _ = lmm.drop_missing_meta(nan, spec, power)
    s2 = LMM(m2, spec).fit(p2).stat
    assert np.isfinite(s2).all() and np.array_equal(s2, LMM(m3, spec).fit(p3).stat)
    num = inf.assign(Subject=np.repeat(np.arange(8.0), 9))
    num.loc[6, "Subject"] = np.inf                    # a numeric subject ID of Inf is missing too
    with pytest.warns(lmm.LMMMetaNaNRows):
        _, _, d4 = lmm.drop_missing_meta(num, spec, power)
    assert d4.n == 3 and d4.rows.tolist() == [3, 6, 29]


def _t23_meta(seed):
    rng = np.random.default_rng(seed)
    meta, power = long_table(8, 9, 3, rng)
    meta["group"] = meta["group"].map({"G1": "A", "G2": "B"})
    meta["time"] = np.tile(np.arange(9.0), 8)
    meta["cov"] = rng.standard_normal(len(meta))
    return meta, power


def test_unknown_effect_is_refused_with_the_names_of_the_fixed_part(tmp_path):
    """MATLAB T23: an effect naming no coefficient (continuous) or term (factor) is refused from
    the metadata alone, before any fit, with fitlme's names listed."""
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    meta, _ = _t23_meta(231)
    labels = get_montage("egi").labels
    power = 10 + np.random.default_rng(1).standard_normal((len(meta), len(labels)))
    path = _write(tmp_path, meta, power, labels)
    base = _params(path, lmm_meta_cols=["Subject", "group", "DV", "time", "cov"])
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(dict(base, lmm_fixed="POWER + cov", lmm_effect="POWERX"))
    assert e.value.code == "unknownEffect"
    assert str(e.value) == ("Effect 'POWERX' is not a coefficient of the fixed part 'POWER + cov' "
                            "(coefficients: (Intercept), cov, POWER)")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(dict(base, lmm_fixed="POWER*group", lmm_effect="POWER:time",
                                  lmm_effect_type="factor", lmm_perm="within_subject"))
    assert e.value.code == "unknownEffect"
    assert str(e.value) == ("Effect 'POWER:time' is not a term of the fixed part 'POWER*group' "
                            "(terms: group, POWER, group:POWER)")
    # names built without a fit: logical as <var>_1, text levels, the first variable fastest,
    # terms of one degree compared from their last variable (all as fitlme R2025a names them on
    # this design)
    m = meta.copy()
    m["flag"] = np.repeat(np.arange(8) % 4 < 2, 9)
    m["group3"] = np.repeat(["A", "B", "C", "A", "B", "C", "A", "B"], 9)
    m["cond"] = np.tile(["x", "y", "z"], len(m) // 3)
    fx = "POWER*group + group3:cond + flag + cov"
    sp = LMMSpec(dv="DV", fixed=fx, effect="POWER", effect_type="continuous")
    coefs, terms = lmm.fixed_names(m, sp)
    assert coefs == ["(Intercept)", "group_B", "cov", "flag_1", "POWER", "group3_B:cond_y",
                     "group3_C:cond_y", "group3_B:cond_z", "group3_C:cond_z", "group_B:POWER"]
    assert [":".join(t) for t in lmm._terms(
        "x4:x1 + x3:x2 + x3:x1 + x1:x2:x4 + x2:x3:x4 + x4 + x2", ["x1", "x2", "x3", "x4"])] == [
        "x2", "x4", "x1:x3", "x2:x3", "x1:x4", "x1:x2:x4", "x2:x3:x4"]   # LinearFormula's order
    assert terms == ["group", "cov", "flag", "POWER", "group3:cond", "group:POWER"]
    model = LMM(m, sp)
    assert model.names == coefs                       # the fit's own design columns
    X, _ = model.design(np.ones((len(m), 1)))
    want = ((m["group3"] == "C") & (m["cond"] == "y")).to_numpy(float)
    assert np.array_equal(X[0, :, coefs.index("group3_C:cond_y")], want)
    for c in coefs:
        assert lmm.check_effect(m, LMMSpec(dv="DV", fixed=fx, effect=c,
                                           effect_type="continuous")) == c
    with pytest.raises(lmm.LMMInputError) as e:       # not fitlme's order
        lmm.check_effect(m, LMMSpec(dv="DV", fixed=fx, effect="cond_y:group3_B",
                                    effect_type="continuous", perm="within_subject"))
    assert e.value.code == "unknownEffect"
    for t in terms:
        assert lmm.check_effect(m, LMMSpec(dv="DV", fixed=fx, effect=t, effect_type="factor",
                                           perm="within_subject")) == t
    assert lmm.check_effect(m, LMMSpec(dv="DV", fixed=fx, effect="POWER:group",
                                       effect_type="factor")) == "group:POWER"


def _not_permuted(effect):
    return ("The group_label scheme relabels subjects between levels of 'group', but the effect "
            f"'{effect}' does not contain 'group', so every permuted map would equal the observed "
            "one. Use within_subject for an effect that varies within subjects, or test an effect "
            "of the group column.")


def test_group_label_needs_the_group_column_or_one_of_its_levels(tmp_path):
    """MATLAB T24: a token names the group only as the group column or one of its coefficients;
    another column (group_size) is refused as effectNotPermuted, a level not in the data (group_C)
    as unknownEffect."""
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    meta, _ = _t23_meta(241)
    meta["group_size"] = 10 + np.repeat(np.arange(8) % 3, 9)
    labels = get_montage("egi").labels
    power = 10 + np.random.default_rng(2).standard_normal((len(meta), len(labels)))
    path = _write(tmp_path, meta, power, labels)
    p = _params(path, lmm_meta_cols=["Subject", "group", "DV", "time", "group_size"],
                lmm_dv="POWER", lmm_fixed="group + group_size", lmm_effect="group_size",
                lmm_perm="group_label")
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(p)
    assert e.value.code == "effectNotPermuted" and str(e.value) == _not_permuted("group_size")
    # a group coefficient for a level not in the data is an unknown effect, listing the real names
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(dict(p, lmm_effect="group_C"))
    assert e.value.code == "unknownEffect"
    assert str(e.value) == ("Effect 'group_C' is not a coefficient of the fixed part "
                            "'group + group_size' (coefficients: (Intercept), group_B, group_size)")
    sp = LMMSpec(dv="POWER", fixed="group + group_size", effect="group_B",
                 effect_type="continuous", perm="group_label")
    assert lmm.check_effect(meta, sp) == "group_B"
    sp = LMMSpec(dv="DV", fixed="POWER*group", effect="group_B:POWER", effect_type="continuous",
                 perm="group_label")
    assert lmm.check_effect(meta, sp) == "group_B:POWER"
    for eff, want in (("group", "group"), ("POWER:group", "group:POWER")):
        sp = LMMSpec(dv="DV", fixed="POWER*group", effect=eff, effect_type="factor")
        assert lmm.check_effect(meta, sp) == want
    # before the file is read only the coarse form can be checked: group_size passes it
    lmm.check_design(LMMSpec(dv="DV", fixed="POWER*group", effect="group_size",
                             effect_type="factor"))
    # the pipeline carries fitlme's term name on
    out = lmm_pipeline.prepare(dict(p, lmm_dv="DV", lmm_fixed="POWER*group",
                                    lmm_effect="POWER:group", lmm_effect_type="factor"))
    assert out[0].effect == "group:POWER"


def ws_fixture(seed, n_ch):
    """MATLAB ws_fixture: 12 subjects (6 per group), 10 trials, 'time' varying within subject,
    a subject-level 'age', response 'sleepDepth' driven by channel 0."""
    rng = np.random.default_rng(seed)
    n_s, n_t = 12, 10
    si = np.repeat(np.arange(n_s), n_t)
    power = rng.standard_normal((si.size, n_ch))
    dv = rng.standard_normal(n_s)[si] + 0.8 * power[:, 0] + rng.standard_normal(si.size)
    meta = pd.DataFrame({"Subject": [f"sub{s + 1:02d}" for s in si],
                         "group": np.where(si < 6, "B", "A"), "age": 21.0 + si,
                         "time": np.tile(np.arange(1.0, n_t + 1), n_s), "sleepDepth": dv})
    return meta, power


def _ws_params(path, **kw):
    p = {"comparison": "mixedmodel", "data_file": path, "channels": "egi",
         "lmm_meta_cols": ["Subject", "group", "age", "time", "sleepDepth"],
         "lmm_dv": "sleepDepth", "lmm_random": "(1|Subject)", "lmm_subject": "Subject",
         "lmm_group": "group", "permutations": 5, "seed": 1}
    p.update(kw)
    return p


def _eff_const_msg(effect, w):
    return ("The within_subject scheme shuffles 'sleepDepth' within each subject, but the effect "
            f"'{effect}' is constant within each subject (largest within-subject share of "
            f"variation among its variables {w}, threshold 1e-06), so the shuffle leaves the "
            "effect in place and gives no null for it. Use group_label for a between-group "
            "effect, or test an effect that varies within subjects.")


def test_within_subject_refuses_an_effect_constant_within_subject(tmp_path):
    """MATLAB T25: the within-subject shuffle leaves each subject's mean of the DV in place, so
    an effect whose variables are all constant within subject has no null under it."""
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    labels = get_montage("egi").labels
    meta, power = ws_fixture(251, len(labels))
    path = _write(tmp_path, meta, power, labels)
    base = _ws_params(path, lmm_perm="within_subject")
    for extra, eff in (({"lmm_fixed": "POWER + group", "lmm_effect": "group",
                         "lmm_effect_type": "factor"}, "group"),
                       ({"lmm_fixed": "POWER + group", "lmm_effect": "group_B",
                         "lmm_effect_type": "continuous"}, "group_B"),
                       ({"lmm_fixed": "POWER + age", "lmm_effect": "age",
                         "lmm_effect_type": "continuous"}, "age")):
        with pytest.raises(lmm.LMMInputError) as e:
            lmm_pipeline.prepare(dict(base, **extra))
        assert e.value.code == "effectConstantWithinSubject"
        assert str(e.value) == _eff_const_msg(eff, 0)
    share = lmm.effect_within_share
    assert share(meta, "time", "Subject") > 1e-6
    assert share(meta, "group:time", "Subject") > 1e-6
    assert share(meta, "group_B:time", "Subject") > 1e-6
    assert np.isnan(share(meta, "group:POWER", "Subject"))
    assert np.isnan(share(meta, "nothing_here", "Subject"))
    out = lmm_pipeline.prepare(dict(base, lmm_fixed="POWER + group*time",
                                    lmm_effect="group_B:time", lmm_effect_type="continuous"))
    assert out[0].effect == "group_B:time"
    m = meta.copy()
    m.loc[4, "group"] = "A"                            # subject 1 is B: one differing trial
    assert share(m, "group", "Subject") > 1e-6
    # the share is printed as MATLAB's %.3g
    from hdeeg_snpm.lmm_pipeline import _g3
    assert [_g3(v) for v in (0.0, 1.23456e-7, 5e-7)] == ["0", "1.23e-07", "5e-07"]


def test_group_label_keeps_a_subject_constant_power_and_excludes_a_flat_one(tmp_path):
    """MATLAB T26: under group_label with a metadata response, relabelling subjects moves the
    response and POWER together, so a POWER constant within subject is a subject-level covariate
    and the channel is kept; a channel with one value on every trial is excluded with its reason.
    MATLAB's third case, a random POWER slope by subject, is refused here (only (1|Subject) is
    fitted)."""
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    labels = get_montage("egi").labels
    meta, power = ws_fixture(261, len(labels))
    subj = meta["Subject"].to_numpy()
    power[:, 39] = pd.Series(power[:, 6]).groupby(subj).transform("mean").to_numpy()
    power[:, 40] = 5.0
    path = _write(tmp_path, meta, power, labels)
    p = _ws_params(path, lmm_fixed="group*POWER", lmm_effect="group", lmm_effect_type="factor",
                   permutations=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = lmm_pipeline.run_lmm(p)
    assert res.excluded.tolist() == [40] and res.excluded_reasons == ["same value in every subject"]
    assert np.isnan(res.T.real_T[40]) and np.isfinite(res.T.real_T[39])
    spec = LMMSpec(dv="sleepDepth", fixed="group*POWER", effect="group", effect_type="factor")
    ref = LMM(res.meta, spec).fit(res.power[:, [39]]).stat[0]
    assert res.T.real_T[39] == pytest.approx(ref, rel=1e-8)   # optimiser tolerance
    with pytest.raises(lmm.UnsupportedRandomEffects):
        lmm_pipeline.prepare(dict(p, lmm_random="(1 + POWER|Subject)"))


def test_group_label_refuses_a_response_constant_within_subject(tmp_path):
    """MATLAB T27: a metadata response constant within every subject makes every fit perfect."""
    from hdeeg_snpm import lmm_pipeline
    from hdeeg_snpm.montage import get_montage

    labels = get_montage("egi").labels
    meta, power = ws_fixture(271, len(labels))
    meta["sleepDepth"] = meta["age"]
    path = _write(tmp_path, meta, power, labels)
    with pytest.raises(lmm.LMMInputError) as e:
        lmm_pipeline.prepare(_ws_params(path, lmm_fixed="group*POWER", lmm_effect="group",
                                        lmm_effect_type="factor"))
    assert e.value.code == "dvConstantWithinSubject"
    assert str(e.value) == (
        "The response 'sleepDepth' is constant within each subject (within-subject share of its "
        "variation 0, threshold 1e-06), so every mixed-model fit is a perfect fit and no "
        "statistic is defined. Analyse subject means with anova1/unpairedT.")


def test_level_coding_follows_snpm_lmm_categorize():
    """MATLAB T23 (second half): a text column that is not converted to categorical keeps its
    levels in order of first appearance, so the first-appearing level is the reference and the
    statistic is that of its own dummy; a converted logical group is named group_true."""
    rng = np.random.default_rng(232)
    meta, power = long_table(10, 6, 2, rng)
    meta["cond"] = np.tile(["pre", "post"], len(meta) // 2)       # 'pre' first, not alphabetical
    meta["DV"] = meta["DV"] + 0.7 * (meta["cond"] == "post")
    spec = LMMSpec(dv="DV", fixed="POWER + cond", effect="cond_post", effect_type="continuous")
    coefs, _ = lmm.fixed_names(meta, spec)
    assert coefs == ["(Intercept)", "cond_post", "POWER"]
    assert lmm.check_effect(meta, spec) == "cond_post"
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_effect(meta, LMMSpec(dv="DV", fixed="POWER + cond", effect="cond_pre",
                                       effect_type="continuous"))
    assert e.value.code == "unknownEffect"
    # the reference changes the statistic: the t of cond_post is the t of a 0/1 'post' column
    model = LMM(meta, spec)
    assert model.names == coefs
    t = model.fit(power).stat
    num = meta.assign(post=(meta["cond"] == "post").astype(float))
    ref = LMM(num, LMMSpec(dv="DV", fixed="POWER + post", effect="post",
                           effect_type="continuous")).fit(power).stat
    assert np.allclose(t, ref, rtol=1e-10, atol=0) and (t > 0).all()
    # and an independent fit (statsmodels MixedLM, ML) with 'pre' as the reference
    MixedLM = pytest.importorskip("statsmodels.regression.mixed_linear_model").MixedLM
    fit = model.fit(power)
    y = meta["DV"].to_numpy()
    for j in range(power.shape[1]):
        X = np.column_stack([np.ones(len(meta)), (meta["cond"] == "post").to_numpy(float),
                             power[:, j]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sm = MixedLM(y, X, groups=model.gidx).fit(reml=False, gtol=1e-10)
        tau2 = float(np.asarray(sm.cov_re).ravel()[0])
        beta, cov, _ = dense_gls(y, X, model.gidx, tau2, sm.scale)
        assert fit.beta[j, 1] == pytest.approx(beta[1], rel=1e-5)
        assert t[j] == pytest.approx(beta[1] / np.sqrt(cov[1, 1]), rel=1e-5)
    # converted (in spec.categorical), the same text is sorted: 'post' becomes the reference
    conv = LMMSpec(dv="DV", fixed="POWER + cond", effect="cond_pre", effect_type="continuous",
                   categorical=("Subject", "group", "cond"))
    assert lmm.fixed_names(meta, conv)[0] == ["(Intercept)", "cond_pre", "POWER"]
    assert np.allclose(LMM(meta, conv).fit(power).stat, -ref, rtol=1e-8, atol=0)
    # a logical group, converted as the group always is: group_true, accepted under group_label
    lg = meta.assign(group=meta["group"] == "G2")
    for eff in ("group_true", "group_true:POWER"):
        sp = LMMSpec(dv="DV", fixed="POWER*group", effect=eff, effect_type="continuous",
                     perm="group_label")
        assert lmm.check_effect(lg, sp) == eff
    assert lmm.fixed_names(lg, sp)[0] == ["(Intercept)", "group_true", "POWER",
                                          "group_true:POWER"]
    with pytest.raises(lmm.LMMInputError) as e:
        lmm.check_effect(lg, LMMSpec(dv="DV", fixed="POWER*group", effect="group_1",
                                     effect_type="continuous", perm="group_label"))
    assert e.value.code == "unknownEffect"
    # a logical that is not converted: reference False, named <var>_1
    fl = meta.assign(flag=np.repeat(np.arange(10) % 2 == 0, 6))
    assert lmm.fixed_names(fl, LMMSpec(dv="DV", fixed="POWER + flag", effect="flag_1",
                                       effect_type="continuous"))[0][1] == "flag_1"
