"""Family-wise error rate of every phase 1 analysis on null data.

For each analysis, simulate data with no effect many times, run the permutation test, and count
how often anything is declared significant at 0.05 (any channel after TFCE correction; any
cluster). A valid test rejects in about 5% of simulations. Noise is spatially correlated
(each channel mixed with its neighbours) so that clusters form as they do in real data.

    python tools/fwe_simulation.py --sims 400 --perms 199 --out _scratch/fwe.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hdeeg_snpm import glm, legacy, missing
from hdeeg_snpm.clusters import ChannelGraph
from hdeeg_snpm.montage import get_montage
from hdeeg_snpm.pipeline import cluster_threshold

ALPHA = 0.05
_M = get_montage("egi")
_G = ChannelGraph(_M.neighbors)
_NCH = len(_M.labels)
_MIX = np.eye(_NCH)
for _i, _row in enumerate(_M.neighbors):
    _MIX[_i, _row[_row >= 0]] = 0.5


def noise(rng, n):
    return rng.standard_normal((n, _NCH)) @ _MIX.T


def _legacy(comparison, x, y, rng, n_perm):
    thr = cluster_threshold(comparison, x.shape[0], y.shape[0])
    rule = missing.channel_rule(comparison, x, y)      # as the pipeline does, once, up front
    x, y = x.copy(), y.copy()
    x[:, ~rule.evaluable] = np.nan
    y[:, ~rule.evaluable] = np.nan
    T = legacy.single_threshold_tfce(x, y, _G, comparison, n_perm=n_perm, rng=rng)
    C = legacy.cluster_analysis(x, y, thr, _G, comparison, n_perm=n_perm, rng=rng)
    return np.nanmin(T.p_correctedTFCE) <= ALPHA, min(c.p for c in C.clusters) <= ALPHA


def _glm(preset, Y, meta, rng, n_perm, role="main", **cols):
    loop = next(lp for lp in glm.plan_preset(preset, Y, meta, **cols) if lp.role == role)
    ctx = glm.fl_context(loop.Y, loop.design)
    ctx = replace(ctx, evaluable=ctx.evaluable & loop.mask)
    r = glm.run_effect(loop.Y, loop.design, _G, n_perm, loop.C, rng, ctx=ctx)
    return np.nanmin(r.p_correctedTFCE) <= ALPHA, min(c.p for c in r.clusters) <= ALPHA


def _long(rng, n_subj, n_cond, groups=None):
    """Repeated measures with a random subject offset; no condition or group effect."""
    subj = np.repeat(np.arange(n_subj), n_cond)
    Y = noise(rng, n_subj * n_cond) + noise(rng, n_subj)[subj]
    meta = pd.DataFrame({"Subject": [f"s{i:02d}" for i in subj],
                         "condition": np.tile([f"c{j}" for j in range(n_cond)], n_subj)})
    if groups:
        meta["group"] = np.array([f"g{i % groups}" for i in range(n_subj)])[subj]
    return Y, meta


def one(args):
    name, seed, n_perm = args
    rng = np.random.default_rng(seed)
    if name == "pairedT":
        return _legacy("pairedT", noise(rng, 20), noise(rng, 20), rng, n_perm)
    if name == "pairedT_missing":       # 20% of cells missing in each condition, at random
        x, y = noise(rng, 20), noise(rng, 20)
        x[rng.random(x.shape) < 0.2] = np.nan
        y[rng.random(y.shape) < 0.2] = np.nan
        return _legacy("pairedT", x, y, rng, n_perm)
    if name == "onesampleT":
        x = noise(rng, 20)
        return _legacy("onesampleT", x, np.zeros_like(x), rng, n_perm)
    if name == "unpairedT":
        return _legacy("unpairedT", noise(rng, 14), noise(rng, 18), rng, n_perm)
    if name == "correlationP_missing":  # scattered missing cells: about a third of channels drop
        x = noise(rng, 24)
        x[rng.integers(0, 24, 70), rng.integers(0, _NCH, 70)] = np.nan
        y = np.repeat(rng.standard_normal((24, 1)), _NCH, axis=1)
        return _legacy("correlationP", x, y, rng, n_perm)
    if name == "unpairedT_missing":     # one group loses 30% of cells in the front half
        x, y = noise(rng, 14), noise(rng, 18)
        y[:, :_NCH // 2][rng.random((18, _NCH // 2)) < 0.3] = np.nan
        return _legacy("unpairedT", x, y, rng, n_perm)
    if name in ("correlationP", "correlationS"):
        x = noise(rng, 24)
        y = np.repeat(rng.standard_normal((24, 1)), _NCH, axis=1)
        return _legacy(name, x, y, rng, n_perm)
    if name == "anova1":
        meta = pd.DataFrame({"group": np.repeat(["a", "b", "c"], 9)})
        return _glm("anova1", noise(rng, 27), meta, rng, n_perm, group_col="group")
    if name == "ancova":                # the covariate has a real effect; the group does not
        age = rng.standard_normal(28)
        meta = pd.DataFrame({"group": np.tile(["a", "b"], 14), "age": age})
        Y = noise(rng, 28) + 1.5 * age[:, None]
        return _glm("ancova", Y, meta, rng, n_perm, group_col="group", covariate_cols=["age"])
    if name == "regression":            # nuisance correlated with the predictor and with the data
        age = rng.standard_normal(28)
        meta = pd.DataFrame({"score": 0.7 * age + rng.standard_normal(28), "age": age})
        Y = noise(rng, 28) + 1.5 * age[:, None]
        return _glm("regression", Y, meta, rng, n_perm, predictor_col="score",
                    covariate_cols=["age"])
    if name == "rmanova":
        Y, meta = _long(rng, 16, 3)
        return _glm("rmanova", Y, meta, rng, n_perm, condition_col="condition",
                    subject_col="Subject")
    if name.startswith("posthoc:"):     # a = b; the third level differs, in mean or in spread
        _, preset, kind = name.split(":")
        if preset == "rmanova":
            Y, meta = _long(rng, 14, 3)
            third, cols = (meta.condition == "c2").to_numpy(), {"condition_col": "condition",
                                                                "subject_col": "Subject"}
            role = "main/posthoc:c0 vs c1"
        else:
            meta = pd.DataFrame({"group": np.tile(["a", "b", "c"], 12),
                                 "age": rng.standard_normal(36)})
            Y = noise(rng, 36) + (0.8 * meta.age.to_numpy()[:, None] if preset == "ancova" else 0)
            third, cols = (meta.group == "c").to_numpy(), {"group_col": "group"}
            if preset == "ancova":
                cols["covariate_cols"] = ["age"]
            role = "main/posthoc:a vs b"
        if kind == "shift":
            Y[third] += 3.0 * Y.std()
        else:
            Y[third] = Y[third].mean(axis=0) + 0.5 * (Y[third] - Y[third].mean(axis=0))
        return _glm(preset, Y, meta, rng, n_perm, role=role, **cols)
    if name == "mixed2way:condition_crossover":   # no condition effect; strong interaction
        Y, meta = _long(rng, 22, 3, groups=2)
        sign = np.where(meta.group == "g0", 1.0, -1.0) * np.where(meta.condition == "c1", 1.0, -1.0)
        Y = Y + 3.0 * (sign * (meta.condition != "c2").to_numpy())[:, None]
        return _glm("mixed2way", Y, meta, rng, n_perm, role="condition", group_col="group",
                    condition_col="condition", subject_col="Subject")
    if name == "mixed2way:group_missing_cells":   # condition effect; one group loses cells
        Y, meta = _long(rng, 20, 2, groups=2)
        Y[(meta.condition == "c1").to_numpy()] += 2.0
        lose = ((meta.group == "g0") & (meta.condition == "c1")).to_numpy()
        lose &= np.isin(meta.Subject, pd.unique(meta.Subject[meta.group == "g0"])[:7])
        Y[np.ix_(lose, np.arange(0, _NCH, 4))] = np.nan
        return _glm("mixed2way", Y, meta, rng, n_perm, role="group", group_col="group",
                    condition_col="condition", subject_col="Subject")
    role = name.split(":")[1]
    Y, meta = _long(rng, 20, 2, groups=2)
    return _glm("mixed2way", Y, meta, rng, n_perm, role=role, group_col="group",
                condition_col="condition", subject_col="Subject")


ANALYSES = ["pairedT", "pairedT_missing", "onesampleT", "unpairedT", "unpairedT_missing",
            "correlationP", "correlationP_missing", "correlationS", "anova1", "ancova", "regression", "rmanova", "mixed2way:group",
            "mixed2way:condition", "mixed2way:interaction",
            "posthoc:anova1:shift", "posthoc:anova1:spread", "posthoc:ancova:shift",
            "posthoc:ancova:spread", "posthoc:rmanova:shift", "posthoc:rmanova:spread",
            "mixed2way:condition_crossover", "mixed2way:group_missing_cells"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--perms", type=int, default=199)
    ap.add_argument("--only", nargs="*", default=ANALYSES)
    ap.add_argument("--out", default="_scratch/fwe.csv")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="add to every simulation seed, for an independent repeat")
    a = ap.parse_args(argv)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    half = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / a.sims)
    rows = []
    with ProcessPoolExecutor(a.workers) as pool:
        for k, name in enumerate(a.only):
            jobs = [(name, a.seed_offset + 100000 * (ANALYSES.index(name) + 1) + s, a.perms)
                    for s in range(a.sims)]
            hits = np.array(list(pool.map(one, jobs, chunksize=4)))
            tfce, clus = hits.mean(axis=0)
            rows.append({"analysis": name, "sims": a.sims, "perms": a.perms,
                         "fwe_tfce": round(float(tfce), 4), "fwe_cluster": round(float(clus), 4),
                         "accept_low": round(ALPHA - half, 4), "accept_high": round(ALPHA + half, 4)})
            print(f"{name:24s} TFCE {tfce:.3f}  cluster {clus:.3f}   "
                  f"(95% range for a true 0.05: {ALPHA - half:.3f} to {ALPHA + half:.3f})", flush=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
