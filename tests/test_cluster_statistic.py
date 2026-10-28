"""The cluster statistic option of the GLM, circular and mixed-model tiers ('sum' or 'mean').

Mirrors the tier MATLAB adds to test_glm_snpm, test_circ_snpm and test_lmm_snpm: the same
permutations under both rules give identical TFCE maps and p, every observed cluster's mean mass
times its size is its summed mass, a broad moderate cluster beats a focal one only under 'sum',
anything else is refused, and the rule in force is recorded in every output file.
"""

import json
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import circ_pipeline, cluster_stat, glm, lmm, lmm_pipeline, pipeline
from hdeeg_snpm.lmm import LMMSpec
from hdeeg_snpm.montage import get_montage
from hdeeg_snpm.output import run_and_write
from test_circ_pipeline import group_params
from test_glm import MIXED, _mixed
from test_lmm import cluster_of, long_table

DIVIDED = "divided by the number of channels"


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


def _same_tfce(a, b, what):
    for f in ("real_T", "real_TFCE", "tMaxTFCE", "p_correctedTFCE", "p_real", "critical_T_TFCE"):
        assert np.array_equal(getattr(a, f), getattr(b, f), equal_nan=True), f"{what}: {f}"


def _sum_is_mean_times_size(mean_clusters, sum_clusters, what):
    mean = {frozenset(c.channels.tolist()): c.mass for c in mean_clusters if c.channels.size}
    summed = {frozenset(c.channels.tolist()): c.mass for c in sum_clusters if c.channels.size}
    assert mean.keys() == summed.keys(), f"{what}: the clusters themselves must not change"
    for ch, m in mean.items():
        assert summed[ch] == pytest.approx(m * len(ch), rel=1e-12), what
    return len(mean)


# --------------------------------------------------------------------------- the rule itself

def test_rule_is_sum_by_default_and_case_insensitive():
    assert cluster_stat.resolve(None) == cluster_stat.resolve("") == "sum"
    assert cluster_stat.resolve("   ") == "sum" and cluster_stat.resolve("MEAN") == "mean"
    assert cluster_stat.resolve("Mean") == "mean" and cluster_stat.resolve(" SUM ") == "sum"
    w, ch = np.array([4.0, np.nan, 9.0, 1.0]), np.array([0, 1, 2])
    assert cluster_stat.score(w, ch, "sum") == 13.0
    assert cluster_stat.score(w, ch, "mean") == 13.0 / 3


@pytest.mark.parametrize("bad, shown", [("bogus", "bogus"), (" Max ", " Max "),
                                        (3, "<non-text value>"), (["sum"], "<non-text value>")])
def test_anything_else_is_refused_with_matlabs_text(bad, shown):
    with pytest.raises(cluster_stat.BadClusterStatistic) as err:
        cluster_stat.resolve(bad)
    assert str(err.value) == f"cluster_statistic must be 'sum' or 'mean'; got '{shown}'."
    assert err.value.code == "badClusterStatistic"


def test_wordings_are_matlabs():
    assert cluster_stat.results_line("sum") == "Cluster statistic: sum of t² (or F) over the cluster"
    assert cluster_stat.results_line("mean") == (
        "Cluster statistic: sum of t² (or F) over the cluster, divided by the number of channels "
        "(mean; reproduces runs before this change)")
    helfrich = " This differs from the FieldTrip maxsize (cluster extent) used by Helfrich et al."
    assert cluster_stat.circ_results_line("sum") == (
        "Cluster statistic: sum of the statistic over the cluster." + helfrich)
    assert cluster_stat.circ_results_line("mean") == (
        "Cluster statistic: sum of the statistic over the cluster, divided by the number of "
        "channels (mean; reproduces runs before this change)." + helfrich)
    assert cluster_stat.circ_recorded("sum") == "sum of the statistic over the cluster"
    assert cluster_stat.card_value("sum") == "Sum of t² (or F) over the cluster"
    assert cluster_stat.card_value("sum", cluster_stat.CIRC_WHAT) == (
        "Sum of the statistic over the cluster")
    assert cluster_stat.card_value("mean") == (
        "Sum of t² (or F) over the cluster, divided by the number of channels (mean; reproduces "
        "runs before this change)")


@pytest.mark.parametrize("comparison", ["anova1", "circ_phase_group", "mixedmodel"])
def test_pipeline_refuses_before_reading_any_file(comparison):
    """The mass tiers check the value first, as MATLAB's do; the t-tests ignore it (below)."""
    with pytest.raises(cluster_stat.BadClusterStatistic, match="'sum' or 'mean'"):
        pipeline.run({"comparison": comparison, "cluster_statistic": "bogus",
                      "data_file": "/nonexistent.csv", "data1_file": "/nonexistent.csv"})


def test_each_tier_refuses_when_called_directly():
    bad = {"comparison": "anova1", "cluster_statistic": "bogus", "data_file": "/nonexistent.csv"}
    for run in (pipeline.run_glm, circ_pipeline.run_circ, lmm_pipeline.run_lmm):
        with pytest.raises(cluster_stat.BadClusterStatistic):
            run(bad)


# --------------------------------------------------------------------------- GLM

def test_glm_every_loop_same_tfce_and_sum_is_mean_times_size(egi_graph):
    """rmanova with post-hoc pairs and mixed2way with its three effects, post-hoc pairs and simple
    effects: the same draws under both rules."""
    rng = np.random.default_rng(3)
    subj = np.repeat(np.arange(12), 3)
    meta = pd.DataFrame({"Subject": subj, "cond": np.tile(["a", "b", "c"], 12)})
    Y = rng.standard_normal((36, egi_graph.n)) + rng.standard_normal(12)[subj, None]
    Y[np.asarray(meta.cond == "c")[:, None] & np.isin(np.arange(egi_graph.n),
                                                     cluster_of(egi_graph))[None, :]] += 1.5
    m2, Y2 = _mixed(np.random.default_rng(4), egi_graph.n, conds=("pre", "mid", "post"))
    n_clusters = 0
    for preset, data, d, cols in (("rmanova", Y, meta, {"subject_col": "Subject",
                                                         "condition_col": "cond"}),
                                  ("mixed2way", Y2, m2, MIXED)):
        runs = {rule: glm.run_preset(preset, data, d, egi_graph, n_perm=60,
                                     rng=np.random.default_rng(9), cluster_statistic=rule, **cols)
                for rule in ("sum", "mean")}
        assert any("/posthoc:" in lp.role for lp in runs["sum"])
        for a, b in zip(runs["mean"], runs["sum"]):
            assert a.role == b.role
            assert (a.result.cluster_statistic, b.result.cluster_statistic) == ("mean", "sum")
            _same_tfce(a.result, b.result, f"{preset} {a.role}")
            n_clusters += _sum_is_mean_times_size(a.result.clusters, b.result.clusters,
                                                  f"{preset} {a.role}")
    assert n_clusters > 0


def test_a_broad_moderate_cluster_beats_a_focal_one_only_under_sum(egi_graph):
    def ring(c, k):
        s = {c}
        for _ in range(k):
            s |= {j for i in s for j in egi_graph.neighbor_lists[i]}
        return sorted(s)

    broad, focal = ring(23, 2), 120                    # 19 channels; 120 is far from them
    assert not set(ring(focal, 1)) & set(ring(23, 3))
    rng = np.random.default_rng(2)
    meta = pd.DataFrame({"group": np.repeat(["a", "b"], 20)})
    Y = rng.standard_normal((40, egi_graph.n))
    Y[20:, broad] += 0.9
    Y[20:, focal] += 3.0
    d = glm.build_design("anova1", meta, group_col="group")
    res = {rule: glm.run_effect(Y, d, egi_graph, n_perm=199, rng=np.random.default_rng(102),
                                cluster_statistic=rule) for rule in ("sum", "mean")}

    def pick(r):
        b = max((c for c in r.clusters if set(broad) & set(c.channels.tolist())),
                key=lambda c: c.channels.size)
        return b, next(c for c in r.clusters if focal in c.channels)

    (bs, fs), (bm, fm) = pick(res["sum"]), pick(res["mean"])
    assert bs.mass > fs.mass and bs.p <= 0.05
    assert fm.mass > bm.mass and bm.p > 0.05
    _same_tfce(res["sum"], res["mean"], "broad vs focal")


# --------------------------------------------------------------------------- circular

def test_circular_same_tfce_and_sum_is_mean_times_size():
    r = {rule: quiet(circ_pipeline.run_circ, group_params(cluster_statistic=rule))
         for rule in ("sum", "mean")}
    assert (r["sum"].cluster_statistic, r["mean"].cluster_statistic) == ("sum", "mean")
    assert np.array_equal(r["sum"].result.perms, r["mean"].result.perms)
    _same_tfce(r["sum"].result, r["mean"].result, "circular")
    assert _sum_is_mean_times_size(r["mean"].result.clusters, r["sum"].result.clusters,
                                   "circular") > 0


# --------------------------------------------------------------------------- mixed model

def test_mixed_model_sum_is_mean_times_size(egi_graph):
    rng = np.random.default_rng(20)
    meta, power = long_table(12, 10, egi_graph.n, rng, slope=1.0, cluster=cluster_of(egi_graph))
    spec = LMMSpec(dv="DV", fixed="POWER", effect="POWER", effect_type="continuous")
    c = {rule: lmm.lmm_cluster(power, meta, spec, egi_graph, n_perm=30,
                               rng=np.random.default_rng(5), cluster_statistic=rule)
         for rule in ("sum", "mean")}
    assert np.array_equal(c["sum"].perms, c["mean"].perms)
    assert c["sum"].cluster_statistic == "sum" and c["mean"].cluster_statistic == "mean"
    assert _sum_is_mean_times_size(c["mean"].clusters, c["sum"].clusters, "mixed model") > 0
    best = min(c["sum"].clusters, key=lambda k: k.p)
    wald = lmm.LMM(meta, spec).fit(power).wald
    assert best.mass == pytest.approx(wald[best.channels].sum())


# --------------------------------------------------------------------------- recorded everywhere

def _rmanova_file(tmp_path):
    labels = list(get_montage("egi").labels)
    rng = np.random.default_rng(21)
    rows = [(f"sub{i:03d}", c) for i in range(1, 11) for c in ("C1", "C2", "C3")]
    frame = pd.DataFrame(rng.standard_normal((len(rows), len(labels))), columns=labels)
    frame.loc[[i for i, (_, c) in enumerate(rows) if c == "C3"], labels[20:40]] += 1.5
    frame.insert(0, "condition", [c for _, c in rows])
    frame.insert(0, "Subject", [s for s, _ in rows])
    frame.to_csv(tmp_path / "t.csv", index=False)
    return {"comparison": "rmanova", "channels": "egi", "data_file": str(tmp_path / "t.csv"),
            "meta_cols": ["Subject", "condition"], "subject_col": "Subject",
            "condition_col": "condition", "permutations": 30, "seed": 1}


def _lmm_file(tmp_path, egi_graph):
    rng = np.random.default_rng(43)
    meta, power = long_table(10, 8, egi_graph.n, rng, slope=1.0, cluster=cluster_of(egi_graph))
    table = pd.concat([meta, pd.DataFrame(power, columns=get_montage("egi").labels)], axis=1)
    table.to_csv(tmp_path / "long.csv", index=False)
    return {"comparison": "mixedmodel", "data_file": str(tmp_path / "long.csv"),
            "channels": "egi", "lmm_meta_cols": ["Subject", "group", "DV"], "lmm_dv": "DV",
            "lmm_fixed": "POWER", "lmm_random": "(1|Subject)", "lmm_effect": "POWER",
            "lmm_effect_type": "continuous", "permutations": 20, "seed": 5}


@pytest.mark.parametrize("tier", ["glm", "circ", "lmm"])
@pytest.mark.parametrize("rule", [None, "sum", "mean"])
def test_rule_is_recorded_in_every_output_file(tier, rule, tmp_path, egi_graph):
    params = {"glm": lambda: _rmanova_file(tmp_path), "circ": lambda: group_params(permutations=30),
              "lmm": lambda: _lmm_file(tmp_path, egi_graph)}[tier]()
    params["output_path"] = str(tmp_path / "out")
    if rule is not None:
        params["cluster_statistic"] = rule
    expected = rule or "sum"
    paths = quiet(run_and_write, params)

    summary = json.loads(paths["json"].read_text())
    effects = summary["effects"] + [p for e in summary["effects"] for p in e["posthoc"]]
    assert len(effects) > (1 if tier == "glm" else 0)
    assert all(e["cluster_statistic"] == "mass" and e["cluster_statistic_rule"] == expected
               for e in effects)
    if tier == "circ":
        page = paths["html"].read_text()
        assert "{imp" not in page and "Angles were declared in rad with zero convention" in page
        assert not re.search(r"\{[A-Za-z_]\w*(\[|\})", page), "unformatted placeholder in report"
        c = summary["effects"][0]["circular"]
        assert c["cluster_statistic"] == cluster_stat.circ_recorded(expected)
        assert c["cluster_statistic_rule"] == expected
    if tier == "lmm":
        mm = summary["effects"][0]["mixed_model"]
        assert mm["cluster_statistic"] == cluster_stat.card_value(expected)
    line = (cluster_stat.circ_results_line if tier == "circ" else cluster_stat.results_line)
    assert line(expected) in summary["notes"]

    html = paths["html"].read_text()
    what = cluster_stat.CIRC_WHAT if tier == "circ" else cluster_stat.GLM_WHAT
    value = cluster_stat.card_value(expected, what)
    assert f"<tr><th>cluster_statistic</th><td>{value}</td></tr>" in html
    note = (cluster_stat.CIRC_NOTICE[expected] if tier == "circ"
            else cluster_stat.results_line(expected) + ".")
    assert note in html
    assert (DIVIDED in html) == (expected == "mean")

    book = pd.read_excel(paths["xlsx"], sheet_name=None)
    assert list(book["settings"].columns) == list(cluster_stat.SETTINGS_COLUMNS)
    assert book["settings"].iloc[0].tolist() == [expected, 0.05]
    if tier == "lmm":
        mm = book["mixedModel"].set_index("Item")["Value"]
        assert mm["cluster_statistic"] == cluster_stat.card_value(expected)


def test_t_tests_ignore_the_key_keep_cluster_extent_and_write_no_settings_sheet(tmp_path):
    labels = list(get_montage("egi").labels)
    rng = np.random.default_rng(1)
    for k in (1, 2):
        frame = pd.DataFrame(rng.standard_normal((12, len(labels))), columns=labels)
        frame.insert(0, "Subject", [f"s{i:02d}" for i in range(12)])
        frame.to_csv(tmp_path / f"d{k}.csv", index=False)
    paths = quiet(run_and_write, {"comparison": "pairedT", "tail": "both", "channels": "egi",
                                  "data1_file": str(tmp_path / "d1.csv"),
                                  "data2_file": str(tmp_path / "d2.csv"), "permutations": 20,
                                  "seed": 1, "output_path": str(tmp_path / "out"),
                                  "cluster_statistic": "bogus"})
    e = json.loads(paths["json"].read_text())["effects"][0]
    assert e["cluster_statistic"] == "size" and "cluster_statistic_rule" not in e
    assert "settings" not in pd.read_excel(paths["xlsx"], sheet_name=None)
    assert "<th>cluster_statistic</th>" not in paths["html"].read_text()
