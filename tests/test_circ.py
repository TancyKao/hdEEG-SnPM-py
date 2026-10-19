"""Circular statistics (``hdeeg_snpm.circ``) against independent references.

Restates the statistic checks of the MATLAB suite ``test_circ_stats.m`` with the same reference
quantities and tolerances, on numpy-generated synthetic data:

1  Hotelling: classical pooled-covariance form at q = 0; covariate-adjusted form vs an
   independent least-squares reconstruction; per-channel df2 under deletion.
2  Circular-linear: rho vs a transcription of CircStat ``circ_corrcl``; F and p vs an
   independent nested-model F test, with and without covariates; per-channel df2.
3  Watson U^2 vs an O(N^2) transcription of the published algorithm; exact enumerated null vs
   the published asymptotic critical values; the p-value series vs the five published values.
4  Rotation invariance of the three statistics; Tier-1 transforms under rotation and reference.
5  Group symmetry (membership inversion, text labels).
7  Ties: row-order invariance, evidence ordering under jitter, heavy ties.
8  Timing projection and non-negative maps.
9  Tier-1 whole-column NaN, against the Freedman-Lane path.
10 Provenance stamp CSV round trip.
11 Degenerate-input guards.

Section 6 (ClusterEnhancement identity) is ``tests/test_tfce.py``. Further checks below cover what
the MATLAB suite leaves to ``core_snpm_circ.m``: the permutation evaluator equals the observed
statistic, one Freedman-Lane index moves cos and sin together, the count-covariate builder, and
the descriptive panel.
"""

import os
import time
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import linalg, stats

from hdeeg_snpm import circ, glm
from hdeeg_snpm import perm as _perm
from hdeeg_snpm.tfce import tfce


def wrap(a):
    return np.angle(np.exp(1j * a))


def maxabs(a, b):
    return float(np.nanmax(np.abs(np.asarray(a) - np.asarray(b))))


# ---------------------------------------------------------------------------- 1 Hotelling

@pytest.fixture(scope="module")
def hot_data():
    rng = np.random.default_rng(11)
    nA, nB, n_ch = 22, 18, 40
    g = np.r_[np.ones(nA, bool), np.zeros(nB, bool)]
    A = wrap(0.4 + 0.55 * rng.standard_normal((nA + nB, n_ch)))
    A[g] = wrap(A[g] + 0.35)
    return A, g, rng


def test_hotelling_classical_form(hot_data):
    A, g, _ = hot_data
    N, n_ch = A.shape
    r = circ.hotelling(A, g)
    T2ref = np.empty(n_ch)
    for ch in range(n_ch):
        Z1 = np.c_[np.cos(A[g, ch]), np.sin(A[g, ch])]
        Z2 = np.c_[np.cos(A[~g, ch]), np.sin(A[~g, ch])]
        n1, n2 = len(Z1), len(Z2)
        d = Z1.mean(0) - Z2.mean(0)
        Sp = ((n1 - 1) * np.cov(Z1.T) + (n2 - 1) * np.cov(Z2.T)) / (n1 + n2 - 2)
        T2ref[ch] = n1 * n2 / (n1 + n2) * d @ np.linalg.solve(Sp, d)
    Fref = T2ref * (N - 3) / (2 * (N - 2))
    assert maxabs(r.T2, T2ref) < 1e-10
    assert maxabs(r.F, Fref) < 1e-10
    assert r.df1 == 2
    assert np.all(r.df2 == N - 3)
    assert maxabs(r.p, stats.f.sf(r.F, 2, N - 3)) < 1e-14
    assert r.levels == ["false", "true"] and r.n1[0] == 22 and r.n2[0] == 18


def test_hotelling_covariate_adjusted(hot_data):
    A, g, _ = hot_data
    N, n_ch = A.shape
    rng = np.random.default_rng(12)
    q = 2
    Cn = np.c_[rng.standard_normal(N), np.arange(1, N + 1) / N]
    r = circ.hotelling(A, g, Cn)
    X = np.c_[np.ones(N), Cn, g.astype(float)]
    v = np.linalg.inv(X.T @ X)[-1, -1]              # c'(X'X)^-1 c, as fitlm's SE^2 / MSE
    nu = N - np.linalg.matrix_rank(X)
    Fref = np.empty(n_ch)
    for ch in range(n_ch):
        bc = linalg.lstsq(X, np.cos(A[:, ch]))[0]
        bs = linalg.lstsq(X, np.sin(A[:, ch]))[0]
        R = np.c_[np.cos(A[:, ch]) - X @ bc, np.sin(A[:, ch]) - X @ bs]
        S = R.T @ R / nu
        d = np.r_[bc[-1], bs[-1]]
        Fref[ch] = (d @ np.linalg.solve(S, d)) / v * (nu - 1) / (2 * nu)
    assert maxabs(r.F, Fref) < 1e-9
    assert np.all(r.df2 == N - q - 3)
    assert maxabs(r.F, circ.hotelling(A, g).F) > 1e-6        # covariates are live


def test_hotelling_deletion_gives_df2_vector(hot_data):
    A, g, _ = hot_data
    N = A.shape[0]
    Am = A.copy()
    Am[:3, 6] = np.nan
    Am[:5, 11] = np.nan
    r = circ.hotelling(Am, g)
    assert r.df2[6] == N - 3 - 3 and r.df2[11] == N - 3 - 5 and r.df2[0] == N - 3
    assert r.n[6] == N - 3 and r.n[11] == N - 5
    # a deleted channel equals the statistic on its retained rows
    keep = ~np.isnan(Am[:, 11])
    assert maxabs(r.F[11], circ.hotelling(A[keep][:, [11]], g[keep]).F) < 1e-12


# ---------------------------------------------------------------------------- 2 circular-linear

def circ_corrcl_ref(a, x):
    """Transcription of CircStat circ_corrcl (first output)."""
    rxs = np.corrcoef(x, np.sin(a))[0, 1]
    rxc = np.corrcoef(x, np.cos(a))[0, 1]
    rcs = np.corrcoef(np.sin(a), np.cos(a))[0, 1]
    return np.sqrt((rxc ** 2 + rxs ** 2 - 2 * rxc * rxs * rcs) / (1 - rcs ** 2))


def nested_F(y, Zred, Zadd):
    """Independent partial F for columns Zadd added to Zred (both without the response)."""
    Xf = np.c_[Zred, Zadd]
    rss_r = ((y - Zred @ linalg.lstsq(Zred, y)[0]) ** 2).sum()
    rss_f = ((y - Xf @ linalg.lstsq(Xf, y)[0]) ** 2).sum()
    df2 = len(y) - Xf.shape[1]
    F = ((rss_r - rss_f) / Zadd.shape[1]) / (rss_f / df2)
    return F, stats.f.sf(F, Zadd.shape[1], df2)


@pytest.fixture(scope="module")
def cl_data():
    rng = np.random.default_rng(22)
    N, n_ch = 34, 25
    A = wrap(2 * np.pi * rng.random((N, n_ch)))
    y = rng.standard_normal(N) + 1.4 * np.cos(A[:, 2]) - 0.9 * np.sin(A[:, 2])
    return A, y, rng


def test_anglinear_matches_circ_corrcl_and_F(cl_data):
    A, y, _ = cl_data
    N, n_ch = A.shape
    r = circ.corr_anglinear(y, A)
    one = np.ones((N, 1))
    rho_ref = np.array([circ_corrcl_ref(A[:, ch], y) for ch in range(n_ch)])
    FP = np.array([nested_F(y, one, np.c_[np.cos(A[:, ch]), np.sin(A[:, ch])])
                   for ch in range(n_ch)])
    assert maxabs(r.rho, rho_ref) < 1e-10
    assert maxabs(r.F, FP[:, 0]) < 1e-10
    assert maxabs(r.p, FP[:, 1]) < 1e-12
    assert r.df1 == 2 and np.all(r.df2 == N - 3)
    assert maxabs(r.rho ** 2, r.R2) < 1e-12
    assert np.nanargmax(r.F) == 2                       # planted channel


def test_anglinear_with_covariates(cl_data):
    A, y, _ = cl_data
    N, n_ch = A.shape
    rng = np.random.default_rng(23)
    Cn = rng.standard_normal((N, 2))
    r = circ.corr_anglinear(y, A, Cn)
    Zred = np.c_[np.ones(N), Cn]
    Fref = np.array([nested_F(y, Zred, np.c_[np.cos(A[:, ch]), np.sin(A[:, ch])])[0]
                     for ch in range(n_ch)])
    assert maxabs(r.F, Fref) < 1e-9
    assert np.all(r.df2 == N - 3 - 2)


def test_anglinear_deletion_gives_df2_vector(cl_data):
    A, y, _ = cl_data
    N = A.shape[0]
    Am = A.copy()
    Am[:4, 8] = np.nan
    r = circ.corr_anglinear(y, Am)
    assert r.df2[8] == N - 3 - 4 and r.df2[0] == N - 3


# ---------------------------------------------------------------------------- 3 Watson U^2

def watson_u2_ref(a1, a2):
    """Independent O(N^2) transcription of the published two-sample U^2 (Zar ch. 27)."""
    a1, a2 = np.mod(a1, 2 * np.pi), np.mod(a2, 2 * np.pi)
    n1, n2 = len(a1), len(a2)
    n = n1 + n2
    lab = np.r_[np.ones(n1), 2 * np.ones(n2)][np.argsort(np.r_[a1, a2], kind="stable")]
    d = np.zeros(n)
    for i in range(n):
        ai = bi = 0
        for j in range(i + 1):
            if lab[j] == 1:
                ai += 1
            else:
                bi += 1
        d[i] = ai / n1 - bi / n2
    return n1 * n2 / n ** 2 * ((d - d.mean()) ** 2).sum()


def test_watson_vs_reference_loop():
    rng = np.random.default_rng(33)
    worst = 0.0
    for _ in range(200):
        n1, n2 = rng.integers(5, 26, size=2)
        a1 = wrap(2 * np.pi * rng.random(n1))
        a2 = wrap(0.7 + 0.9 * rng.standard_normal(n2))
        g = np.r_[np.ones(n1, bool), np.zeros(n2, bool)]
        u = circ.watson_u2(np.r_[a1, a2], g).U2[0]
        worst = max(worst, abs(u - watson_u2_ref(a1, a2)))
    assert worst < 1e-12


def test_watson_exact_null_on_published_scale():
    rng = np.random.default_rng(34)
    n1 = n2 = 8
    N = n1 + n2
    a = wrap(2 * np.pi * rng.random(N))
    null = []
    for idx in combinations(range(N), n1):
        gk = np.zeros(N, bool)
        gk[list(idx)] = True
        null.append(circ.watson_u2(a, gk).U2[0])
    assert len(null) == 12870
    qexact = np.quantile(null, [0.90, 0.95, 0.99], method="hazen")   # MATLAB quantile
    assert np.max(np.abs(qexact - [0.152, 0.187, 0.268])) < 0.05


def test_watson_pvalue_series():
    uu = np.array([0.152, 0.187, 0.221, 0.268, 0.385])
    tgt = np.array([0.10, 0.05, 0.025, 0.01, 0.001])
    assert np.max(np.abs(circ.watson_u2_pvalue(uu) - tgt)) < 5e-4
    assert np.all(np.diff(circ.watson_u2_pvalue(np.linspace(0.02, 1, 500))) <= 0)
    pp = circ.watson_u2_pvalue([0, 1e-6, 0.001, 5, 50])
    assert np.all((pp >= 0) & (pp <= 1))
    assert np.isnan(circ.watson_u2_pvalue(np.nan))
    assert pp[0] == 1 and pp[1] == 1


# ---------------------------------------------------------------------------- 4 rotation

def test_rotation_invariance():
    rng = np.random.default_rng(44)
    N, n_ch = 36, 30
    g = np.r_[np.ones(18, bool), np.zeros(18, bool)]
    A = wrap(0.3 + 0.45 * rng.standard_normal((N, n_ch)))
    A[g] = wrap(A[g] + 0.3)
    y = rng.standard_normal(N) + 1.2 * np.cos(A[:, 4])
    Ar = wrap(A + 1.9137)

    assert maxabs(circ.hotelling(A, g).F, circ.hotelling(Ar, g).F) < 1e-10
    assert maxabs(circ.watson_u2(A, g).U2, circ.watson_u2(Ar, g).U2) < 1e-10
    assert maxabs(circ.corr_anglinear(y, A).F, circ.corr_anglinear(y, Ar).F) < 1e-10

    Ys1, _ = circ.linearise(A, "signed")
    Ys2, _ = circ.linearise(Ar, "signed")
    Yu1, _ = circ.linearise(A, "unsigned")
    Yu2, _ = circ.linearise(Ar, "unsigned")
    assert maxabs(Ys1, Ys2) < 1e-10               # theta_ref co-rotates
    assert maxabs(Yu1, Yu2) > 0.1                 # a-priori theta0 does not
    assert maxabs(Ys1, wrap(A)) > 0.1             # but depends on the reference direction


# ---------------------------------------------------------------------------- 5 symmetry

def test_group_symmetry():
    rng = np.random.default_rng(55)
    N, n_ch = 30, 25
    g = np.r_[np.ones(14, bool), np.zeros(16, bool)]
    A = wrap(0.2 + 0.7 * rng.standard_normal((N, n_ch)))
    A[g] = wrap(A[g] + 0.4)
    Fa, Fb = circ.hotelling(A, g).F, circ.hotelling(A, ~g).F
    Ua, Ub = circ.watson_u2(A, g).U2, circ.watson_u2(A, ~g).U2
    assert maxabs(Fa, Fb) < 1e-10
    assert maxabs(Ua, Ub) < 1e-12
    gs = np.where(g, "pt", "ctrl")
    assert maxabs(Fa, circ.hotelling(A, gs).F) < 1e-10
    assert maxabs(Ua, circ.watson_u2(A, gs).U2) < 1e-12
    assert circ.hotelling(A, gs).levels == ["ctrl", "pt"]
    gc = pd.Categorical(gs, categories=["pt", "ctrl", "unused"])
    assert maxabs(Fa, circ.hotelling(A, gc).F) < 1e-10
    assert circ.hotelling(A, gc).levels == ["pt", "ctrl"]


# ---------------------------------------------------------------------------- 7 ties

def test_watson_ties():
    rng = np.random.default_rng(77)
    N, n_ch = 40, 60
    g = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    Adeg = np.round(np.degrees(wrap(2 * np.pi * rng.random((N, n_ch)))))
    Aexact = np.radians(Adeg)
    assert sum(N - np.unique(Adeg[:, c]).size for c in range(n_ch)) > 0

    U2a = circ.watson_u2(Aexact, g).U2
    pr = rng.permutation(N)
    assert maxabs(U2a, circ.watson_u2(Aexact[pr], g[pr]).U2) < 1e-14

    # evidence ordering preserved under a 0.01-degree jitter (uncorrected permutation p)
    sub = slice(0, 12)
    Aj = np.radians(Adeg + 0.01 * (2 * rng.random((N, n_ch)) - 1))
    obsT = circ.watson_u2(Aexact[:, sub], g).U2
    obsJ = circ.watson_u2(Aj[:, sub], g).U2
    cntT = np.zeros(12)
    cntJ = np.zeros(12)
    prng = np.random.default_rng(771)
    for _ in range(499):
        gp = g[prng.permutation(N)]
        cntT += circ.watson_u2(Aexact[:, sub], gp).U2 >= obsT
        cntJ += circ.watson_u2(Aj[:, sub], gp).U2 >= obsJ
    rs = stats.spearmanr((cntT + 1) / 500, (cntJ + 1) / 500).statistic
    assert rs > 0.9

    Acoarse = np.radians(30 * rng.integers(0, 12, (N, n_ch)))
    Uc = circ.watson_u2(Acoarse, g).U2
    prc = rng.permutation(N)
    assert maxabs(Uc, circ.watson_u2(Acoarse[prc], g[prc]).U2) < 1e-14
    assert np.all(np.isfinite(Uc)) and np.all(Uc >= 0)


# ---------------------------------------------------------------------------- 8 timing

@pytest.mark.skipif(os.environ.get("CI") == "true", reason="wall-clock bound; shared CI runners "
                    "are too variable for it")
def test_timing_projection(egi_graph):
    rng = np.random.default_rng(88)
    n_ch = egi_graph.n
    N = 40
    g = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    A = wrap(0.3 + 0.5 * rng.standard_normal((N, n_ch)))
    y = rng.standard_normal(N)
    Fmap = circ.hotelling(A, g).F
    Umap = circ.watson_u2(A, g).U2
    assert np.all(Fmap[~np.isnan(Fmap)] >= 0) and np.all(Umap[~np.isnan(Umap)] >= 0)

    def med(fn, k):
        ts = []
        for _ in range(k):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return float(np.median(ts))

    t_h = med(lambda: circ.hotelling(A, g[rng.permutation(N)]), 20)
    t_u = med(lambda: circ.watson_u2(A, g[rng.permutation(N)]), 20)
    t_c = med(lambda: circ.corr_anglinear(y[rng.permutation(N)], A), 20)
    t_tF = med(lambda: tfce(Fmap, egi_graph), 10)
    t_tU = med(lambda: tfce(Umap, egi_graph, dh=0.005), 10)
    worst = 5000 * max(t_h + t_tF, t_u + t_tU, t_c + t_tF)
    # generous: this guards against an order-of-magnitude regression, not a few seconds
    assert worst < 180, f"projected 5000-permutation run {worst:.1f} s"


# ---------------------------------------------------------------------------- 9 whole column

def test_linearise_whole_column_nan_vs_freedman_lane():
    rng = np.random.default_rng(99)
    N, n_ch = 40, 178
    A = wrap(0.25 + 0.45 * rng.standard_normal((N, n_ch)))
    A[6, 8] = np.nan
    Y, info = circ.linearise(A, "signed")
    assert np.all(np.isnan(Y[:, 8])) and not info["valid"][8]
    assert info["reason"][8] == "missing_subjects=1"
    assert np.all(np.isfinite(np.delete(Y, 8, axis=1)))

    X = np.c_[np.ones(N), np.r_[np.ones(20), np.zeros(20)]]
    C = np.array([[0.0, 1.0]])
    design = glm.Design(X, C, np.array([True, False]), np.ones(N), "free", "")
    pe = rng.permutation(N)

    def observed_and_once(Yin):
        ctx = glm.fl_context(Yin, design)
        return (glm.glm_stat(Yin, X, C)[0], glm.glm_stat(ctx.fit + ctx.resid[pe], X, C)[0])

    ref = np.angle(np.nansum(np.exp(1j * A), axis=0))
    Yraw = wrap(A - ref)                                       # per-cell NaN: the wrong way
    s_raw, p_raw = observed_and_once(Yraw)
    assert np.isfinite(s_raw[8]) and np.isnan(p_raw[8])        # the defect being prevented
    s_col, p_col = observed_and_once(Y)
    assert np.isnan(s_col[8]) and np.isnan(p_col[8])
    others = np.arange(n_ch) != 8
    assert np.all(np.isfinite(s_col[others])) and np.all(np.isfinite(p_col[others]))

    Ag = wrap(0.3 + 0.2 * rng.standard_normal((N, 6)))
    Ag[:, 1] = wrap(2 * np.pi * rng.random(N))
    Ag[0, 2] = wrap(Ag[0, 2] + np.pi)
    Yg, ig = circ.linearise(Ag, "signed")
    assert np.all(np.isnan(Yg[:, 1])) and "low_concentration" in ig["reason"][1]
    assert np.all(np.isnan(Yg[:, 2])) and "wraparound" in ig["reason"][2]


# ---------------------------------------------------------------------------- 10 stamp

def test_stamp_csv_round_trip(tmp_path):
    rng = np.random.default_rng(101)
    N, n_ch = 12, 5
    A = wrap(0.2 + 0.3 * rng.standard_normal((N, n_ch)))
    labels = [f"E{k + 1}" for k in range(n_ch)]
    Yu, infoU = circ.linearise(A, "unsigned", count_file="spindle_counts.csv",
                               convention="SO_phase_ccw_updown")
    T = circ.linearise_table(Yu, infoU, labels, np.arange(1, N + 1))
    s = infoU["stamp_string"]
    assert not any(c in s for c in ',"\n\r')
    assert infoU["stamp"]["requires_count_covariate"] == "1"
    assert infoU["stamp"]["count_file"] == "spindle_counts.csv"
    assert "circ_provenance" in T.columns and len(T) == N and T["circ_provenance"].nunique() == 1
    assert T["Subject"].iloc[0] == "sub001"

    path = tmp_path / "t.csv"
    T.to_csv(path, index=False)
    T2 = pd.read_csv(path)
    assert circ.decode_stamp(T2["circ_provenance"].iloc[0]) == infoU["stamp"]
    assert np.max(np.abs(T[labels].to_numpy() - T2[labels].to_numpy())) < 1e-9

    _, infoS = circ.linearise(A, "signed")
    assert infoS["stamp"]["requires_count_covariate"] == "0"
    assert infoS["stamp"]["measure"] == "circ_signed_deviation"
    assert infoU["stamp"]["measure"] == "circ_unsigned_distance"
    assert infoU["stamp"]["rotation"] == "theta0_0"


# ---------------------------------------------------------------------------- 11 degenerate

def test_degenerate_guards():
    rng = np.random.default_rng(111)
    N, n_ch = 20, 4
    g = np.r_[np.ones(10, bool), np.zeros(10, bool)]
    A = wrap(0.3 + 0.4 * rng.standard_normal((N, n_ch)))
    A[:, 1] = 0.7

    r = circ.hotelling(A, g)
    assert np.isnan(r.F[1]) and "singular" in r.reason[1]
    assert np.all(np.isfinite(r.F[[0, 2, 3]]))

    c = circ.corr_anglinear(rng.standard_normal(N), A)
    assert np.isnan(c.F[1]) and np.isnan(c.rho[1]) and c.reason[1]

    assert abs(circ.watson_u2(A, g).U2[1]) < 1e-14

    Am = A.copy()
    Am[:, 2] = np.nan
    rm = circ.hotelling(Am, g)
    assert np.isnan(rm.F[2]) and rm.reason[2] == "no observations"

    As = A.copy()
    As[g, 3] = np.nan
    assert np.isnan(circ.hotelling(As, g).F[3]) and np.isnan(circ.watson_u2(As, g).U2[3])

    assert np.isnan(circ.hotelling(A, g, valid_channels=[1, 0, 1, 1]).F[1])
    assert np.isfinite(circ.hotelling(A, g, valid_channels=[1, 0, 1, 1]).F[0])
    assert np.all(np.isnan(circ.hotelling(A, g, min_per_group=11).F))
    with pytest.raises(circ.CircInputError):
        circ.hotelling(A, np.arange(N) % 3)


# ---------------------------------------------------------------------------- permutation side

def test_perm_evaluator_equals_observed_and_moves_cos_sin_together():
    """Plain channels: HotellingPerm on the observed embedding equals ``hotelling``; one
    Freedman-Lane index under the intercept-only nuisance equals relabelling the subjects."""
    rng = np.random.default_rng(5)
    N, n_ch = 30, 12
    gi = np.r_[np.ones(14), np.zeros(16)]
    A = wrap(0.4 + 0.6 * rng.standard_normal((N, n_ch)))
    A[3, 4] = np.nan                                           # rides along, deleted per channel
    valid = np.ones(n_ch, bool)
    hp = circ.HotellingPerm(gi, None, np.zeros(n_ch, bool), valid)
    F, p, df2 = hp.evaluate(np.cos(A), np.sin(A))
    ref = circ.hotelling(A, gi)
    assert maxabs(F, ref.F) < 1e-12 and maxabs(p, ref.p) < 1e-12
    assert np.array_equal(df2, ref.df2)

    ctx = circ.fl_context(A, None, np.zeros(n_ch, bool), valid)
    assert ctx.fit.shape == (N, 2 * n_ch) and ctx.perm_type == "free"
    yc0, ys0 = circ.permuted_embedding(ctx, np.arange(N))
    assert maxabs(yc0, np.cos(A)) < 1e-14 and maxabs(ys0, np.sin(A)) < 1e-14
    perms = _perm.freedman_lane_indices(N, 5, "free", ctx.blocks, rng)
    for pe in perms:
        Fp, _, _ = hp.evaluate(*circ.permuted_embedding(ctx, pe))
        assert maxabs(Fp, circ.hotelling(A[pe], gi).F) < 1e-12


def test_perm_evaluator_with_per_channel_covariate():
    rng = np.random.default_rng(6)
    N, n_ch = 32, 10
    gi = np.r_[np.ones(15), np.zeros(17)]
    A = wrap(0.4 + 0.6 * rng.standard_normal((N, n_ch)))
    logC = np.log(rng.integers(20, 200, (N, n_ch)).astype(float))
    usecov = np.ones(n_ch, bool)
    usecov[0] = False
    valid = np.ones(n_ch, bool)
    valid[9] = False
    hp = circ.HotellingPerm(gi, logC, usecov, valid)
    F, p, df2 = hp.evaluate(np.cos(A), np.sin(A))
    for ch in range(1, 9):                       # observed: equals hotelling with that covariate
        r = circ.hotelling(A[:, [ch]], gi, logC[:, ch])
        assert abs(F[ch] - r.F[0]) < 1e-10 and df2[ch] == N - 4
    assert abs(F[0] - circ.hotelling(A[:, [0]], gi).F[0]) < 1e-12
    assert np.isnan(F[9]) and np.isnan(p[9])

    ctx = circ.fl_context(A, logC, usecov, valid)
    pe = rng.permutation(N)
    yc, ys = circ.permuted_embedding(ctx, pe)
    Fp, _, _ = hp.evaluate(yc, ys)
    for ch in range(1, 9):                       # independent least squares on the embedding
        X = np.c_[np.ones(N), logC[:, ch], gi]
        Z = np.c_[yc[:, ch], ys[:, ch]]
        B = linalg.lstsq(X, Z)[0]
        S = (Z - X @ B).T @ (Z - X @ B) / (N - 3)
        d = B[-1]
        ref = d @ np.linalg.solve(S, d) / np.linalg.inv(X.T @ X)[-1, -1] * (N - 4) / (2 * (N - 3))
        assert abs(Fp[ch] - ref) < 1e-9
        # FL with this nuisance: the fit is the per-channel line on logC, the residuals move
        for y, col in ((np.cos(A[:, ch]), yc[:, ch]), (np.sin(A[:, ch]), ys[:, ch])):
            Zn = np.c_[np.ones(N), logC[:, ch]]
            fit = Zn @ linalg.lstsq(Zn, y)[0]
            assert maxabs(col, fit + (y - fit)[pe]) < 1e-12
    assert np.all(np.isnan(yc[:, 9]))


# ---------------------------------------------------------------------------- count covariate

def test_count_covariate():
    rng = np.random.default_rng(7)
    N, n_ch = 30, 12
    gi = np.r_[np.ones(15), np.zeros(15)]
    A = wrap(rng.standard_normal((N, n_ch)))
    C = rng.integers(20, 200, (N, n_ch)).astype(float)
    C[:, 3] = 50                                            # constant -> covariate dropped
    C[:, 5] = np.where(gi == 1, 300, 30) + rng.integers(0, 5, N)   # disjoint -> collinear
    C[2, 7] = 0                                             # no usable count -> angle dropped
    with pytest.warns(UserWarning, match="no usable event count"):
        cc = circ.count_covariate(A, gi, C)
    assert np.isnan(cc.A[2, 7]) and np.isnan(cc.logC[2, 7]) and np.isfinite(A[2, 7])
    assert not cc.usecov[3] and not cc.collinear[3]
    assert cc.collinear[5] and not cc.usecov[5]
    assert cc.usecov[0]
    for ch in (0, 1, 7):
        ok = np.isfinite(cc.logC[:, ch])
        assert abs(cc.report["pb"][ch] - stats.pearsonr(cc.logC[ok, ch], gi[ok])[0]) < 1e-12
    assert cc.report["n_cells_no_count"] == 1 and "confound diagnostic" in cc.report["diagnostic"]

    cs = circ.count_covariate(A, gi, np.where(C > 0, C, 1), mode="per_subject")
    assert np.allclose(cs.logC, np.log(np.where(C > 0, C, 1).mean(axis=1))[:, None])

    with pytest.raises(circ.CircInputError, match="non-negative integers") as e:
        circ.count_covariate(A, gi, C + 0.5)
    assert e.value.code == "circCountsInvalid"
    Cm = C.copy()
    Cm[:, :2] = np.nan
    with pytest.raises(circ.CircInputError) as e:
        circ.count_covariate(A, gi, Cm)
    assert e.value.code == "circCountsMissing"
    Cc = np.where(C > 0, C, 1)
    Cc[:, :4] = np.where(gi == 1, 300, 30)[:, None] + rng.integers(0, 5, (N, 4))
    with pytest.raises(circ.CircInputError) as e:
        circ.count_covariate(A, gi, Cc)
    assert e.value.code == "circCountCollinear"


# ---------------------------------------------------------------------------- descriptive

def test_descriptive_matches_scipy():
    rng = np.random.default_rng(8)
    N, n_ch = 24, 6
    gi = np.r_[np.ones(12), np.zeros(12)]
    A = wrap(1.0 + 0.5 * rng.standard_normal((N, n_ch)))
    A[0, 0] = np.nan
    ev = np.ones(n_ch, bool)
    ev[5] = False
    d = circ.descriptive(A, gi, ev)
    assert d["n_channels"] == 5 and "Descriptive only" in d["note"]
    for grp, rows in zip(d["group"], (gi == 1, gi == 0)):
        a = A[rows][:, :5].ravel()
        a = a[np.isfinite(a)]
        assert grp["n"] == a.size
        assert abs(grp["mean_deg"] - np.degrees(stats.circmean(a))) < 1e-9
        assert abs(grp["sd_deg"] - np.degrees(stats.circstd(a))) < 1e-9
        assert abs(grp["R"] - abs(np.exp(1j * a).mean())) < 1e-12
        z, n = grp["rayleigh_z"], a.size
        approx = np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n)
                               - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4)
                               / (288 * n ** 2))
        assert abs(grp["rayleigh_p"] - approx) < 1e-3 * max(approx, 1e-12) + 1e-12
    one = circ.descriptive(A)["group"]
    assert len(one) == 1 and one[0]["label"] == "All subjects"
    md = circ.mean_direction_deg(A)
    assert abs(md[0] - np.mod(np.degrees(stats.circmean(A[1:, 0])), 360)) < 1e-9
