"""The circular tier from files to results (``hdeeg_snpm.circ_pipeline``), without MATLAB.

Import and its refusals, the convention check, the locked parameters, the fixed gate, the Tier-1
guard in front of the other tiers, planted-effect recovery with a null-split negative control,
and the four output files. Inputs are the synthetic example files in ``examples/data``.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hdeeg_snpm import circ, circ_pipeline, pipeline
from hdeeg_snpm.output import run_and_write

DATA = Path(__file__).parent.parent / "examples" / "data"


def group_params(**kw):
    p = {"comparison": "circ_phase_group", "channels": "egi",
         "data1_file": str(DATA / "circ_phase_groupA.csv"),
         "data2_file": str(DATA / "circ_phase_groupB.csv"),
         "count1_file": str(DATA / "circ_phase_countsA.csv"),
         "count2_file": str(DATA / "circ_phase_countsB.csv"),
         "circ_units": "rad", "circ_convention": "literature_uppeak0",
         "permutations": 99, "seed": 1}
    p.update(kw)
    return p


def anglinear_params(**kw):
    p = {"comparison": "circ_corrAngLinear", "channels": "egi",
         "data1_file": str(DATA / "circ_anglinear_angles.csv"),
         "data2_file": str(DATA / "circ_anglinear_measure.csv"),
         "circ_units": "rad", "circ_convention": "literature_uppeak0",
         "permutations": 99, "seed": 1}
    p.update(kw)
    return p


def quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------------------- import

def test_import_requires_units_and_convention():
    raw = np.array([[0.1, 1.2], [2.0, np.nan]])
    for units in (None, "", 3):
        with pytest.raises(circ.CircInputError) as e:
            circ.import_angles(raw, units, "literature_uppeak0")
        assert e.value.code == "circUnitsRequired"
    with pytest.raises(circ.CircInputError, match="must be 'rad' or 'deg'; got 'grad'"):
        circ.import_angles(raw, "grad", "literature_uppeak0")
    for conv in (None, "", "upstate"):
        with pytest.raises(circ.CircInputError) as e:
            circ.import_angles(raw, "rad", conv)
        assert e.value.code == "circConventionRequired"
    with pytest.raises(circ.CircInputError, match="finite scalar"):
        circ.import_angles(raw, "rad", "custom")
    with pytest.raises(circ.CircInputError) as e:
        circ.import_angles(raw * 100, "rad", "literature_uppeak0")
    assert e.value.code == "circUnitsOutOfRange"
    with pytest.raises(circ.CircInputError) as e:
        circ.import_angles(np.full((2, 2), np.nan), "rad", "literature_uppeak0")
    assert e.value.code == "circNoAngles"


def test_import_units_rotation_and_wrapping():
    rng = np.random.default_rng(0)
    deg = rng.uniform(0, 360, (5, 4)) + 0.123
    deg[0, 0] = np.nan
    A, info = circ.import_angles(deg, "degrees", "luna_zerocross0")
    expect = np.mod(np.radians(deg + 90), 2 * np.pi)
    assert np.allclose(A, expect, equal_nan=True) and np.isnan(A[0, 0])
    assert info["units"] == "deg" and info["rotation_deg"] == 90 and info["n_finite"] == 19
    A2, info2 = circ.import_angles(np.radians(deg), "rad", "custom", zero_offset_deg=-450)
    assert info2["rotation_deg"] == 270
    assert np.nanmax(np.abs(np.angle(np.exp(1j * (A2 - np.radians(deg - 90)))))) < 1e-12
    A3, _ = circ.import_angles(np.radians(deg), "rad", "turtlewave_pre_v4")
    assert np.nanmax(np.abs(np.angle(np.exp(1j * (A3 - np.radians(deg) - np.pi))))) < 1e-12
    assert np.nanmin(A3) >= 0 and np.nanmax(A3) < 2 * np.pi


def test_whole_degree_refused_for_u2_only():
    deg = np.round(np.random.default_rng(1).uniform(0, 359, (6, 3)))
    with pytest.raises(circ.CircInputError) as e:
        circ.import_angles(deg, "deg", "literature_uppeak0", resolution_gate="error")
    assert e.value.code == "circResolutionTooCoarse" and "whole-degree" in str(e.value)
    with pytest.warns(UserWarning, match="would REJECT"):
        _, info = circ.import_angles(deg, "deg", "literature_uppeak0")
    assert info["whole_degree"] and info["quantised"]


def test_inversion_check():
    rng = np.random.default_rng(2)
    x = np.linspace(-1, 1, 9)
    imp = {"convention": "literature_uppeak0", "rotation_deg": 0.0}
    A = np.mod(np.pi + 0.3 * rng.standard_normal((20, 9)), 2 * np.pi)
    with pytest.raises(circ.CircInputError) as e:
        circ.check_inversion(A, np.ones(9, bool), x, imp)
    assert e.value.code == "circPhaseConventionInverted"
    with pytest.warns(UserWarning, match="large deviation"):
        info = circ.check_inversion(np.mod(A - np.radians(95), 2 * np.pi), np.ones(9, bool), x, imp)
    assert info["verdict"] == "suspect"
    info = circ.check_inversion(np.mod(A - np.pi, 2 * np.pi), np.ones(9, bool), x, imp)
    assert info["verdict"] == "pass" and info["channels"].tolist() == [6, 7, 8]   # anterior third
    assert circ.check_inversion(A, np.zeros(9, bool), x, imp)["verdict"] == "skipped"


# ---------------------------------------------------------------------------- locked params

@pytest.mark.parametrize("key, value, code", [("tail", "left", "circTailNotSupported"),
                                              ("datatype", "logscale", "circDatatypeNotSupported")])
def test_locked_parameters_refused(key, value, code):
    with pytest.raises(circ.CircInputError) as e:
        circ_pipeline.prepare_circ(group_params(**{key: value}))
    assert e.value.code == code
    quiet(circ_pipeline.prepare_circ, group_params(tail="both", datatype="absolute"))


def test_counts_required_and_missing_convention():
    with pytest.raises(circ.CircInputError) as e:
        quiet(circ_pipeline.prepare_circ, group_params(count2_file=""))
    assert e.value.code == "circCountsRequired"
    p = group_params()
    del p["circ_convention"]
    with pytest.raises(circ.CircInputError) as e:
        circ_pipeline.prepare_circ(p)
    assert e.value.code == "circConventionRequired"


def test_retired_keys_name_their_replacement():
    for key in circ_pipeline.RETIRED:
        with pytest.raises(ValueError, match="is RETIRED"):
            pipeline.run(group_params(comparison=key))


def test_u2_refuses_whole_degree_files(tmp_path):
    for g in ("A", "B"):
        t = pd.read_csv(DATA / f"circ_phase_group{g}.csv")
        cols = [c for c in t.columns if c != "Subject"]
        t[cols] = np.round(np.degrees(t[cols]))
        t.to_csv(tmp_path / f"{g}.csv", index=False)
    p = group_params(comparison="circ_phase_group_u2", circ_units="deg",
                     data1_file=str(tmp_path / "A.csv"), data2_file=str(tmp_path / "B.csv"))
    with pytest.raises(circ.CircInputError) as e:
        circ_pipeline.prepare_circ(p)
    assert e.value.code == "circResolutionTooCoarse"
    prep = quiet(circ_pipeline.prepare_circ, dict(p, comparison="circ_phase_group"))
    assert prep.imp["whole_degree"]


# ---------------------------------------------------------------------------- the gate

def test_gate_held_fixed_and_drift_is_loud(monkeypatch):
    res = quiet(circ_pipeline.run_circ, group_params(comparison="circ_phase_group_u2"))
    assert np.array_equal(~np.isnan(res.result.real_T), res.prepared.evaluable)

    real = circ_pipeline.build_engine

    def drifting(prep):
        eng = real(prep)
        base = eng.perm_map

        def perm_map(ix):
            s, p = base(ix)
            s = s.copy()
            s[np.flatnonzero(prep.evaluable)[0]] = np.nan
            return s, p

        eng.perm_map = perm_map
        return eng

    monkeypatch.setattr(circ_pipeline, "build_engine", drifting)
    with pytest.raises(circ_pipeline.CircGateDrift):
        quiet(circ_pipeline.run_circ, group_params())


def test_excluded_channels_have_reasons(tmp_path):
    a = pd.read_csv(DATA / "circ_phase_groupA.csv")
    a.loc[:9, "E20"] = np.nan                         # 6 left in group A: below the gate
    a.loc[0, "E81"] = np.nan                          # incomplete under the count covariate
    a.to_csv(tmp_path / "A.csv", index=False)
    prep = quiet(circ_pipeline.prepare_circ, group_params(data1_file=str(tmp_path / "A.csv")))
    lab = prep.setup.labels
    got = {lab[i]: r for i, r in zip(prep.excluded, prep.excluded_reasons)}
    both = f"{circ_pipeline.REASON_BELOW}; {circ_pipeline.REASON_INCOMPLETE}"
    assert got == {"E20": both, "E81": circ_pipeline.REASON_INCOMPLETE}
    assert any(f"not tested ({both}): E20" in n for n in prep.notes)



def test_anglinear_exclusion_reason_names_no_groups(tmp_path):
    """MATLAB test_circ_snpm T13: circ_corrAngLinear is paired by subject, so its only exclusion
    route is the >=8 gate on subjects with both an angle and a measure value; the reasons say so
    and do not mention groups."""
    a = pd.read_csv(DATA / "circ_anglinear_angles.csv")
    ch = [c for c in a.columns if c.startswith("E")]
    a.loc[: len(a) - 8, "E5"] = np.nan                # 7 subjects left at E5
    a.to_csv(tmp_path / "A.csv", index=False)
    prep = quiet(circ_pipeline.prepare_circ, anglinear_params(data1_file=str(tmp_path / "A.csv")))
    lab = prep.setup.labels
    got = {lab[i]: r for i, r in zip(prep.excluded, prep.excluded_reasons)}
    assert "E5" in ch and got["E5"] == "fewer than 8 subjects with both an angle and a measure value"
    assert all("group" not in r for r in got.values())
    assert circ_pipeline.exclusion_reason("circ_corrAngLinear") == (
        "not estimable: fewer than 8 subjects with both an angle and a measure value")
    assert circ_pipeline.exclusion_reason("circ_phase_group") == (
        "not estimable: fewer than 8 subjects per group, incomplete column under a per-channel "
        "covariate, or log(count) collinear with group")


# ---------------------------------------------------------------------------- recovery

def test_planted_shift_recovered_and_null_split_quiet(tmp_path):
    res = quiet(circ_pipeline.run_circ, group_params(permutations=199))
    lab = res.setup.labels
    assert "E129" in [lab[i] for i in res.correctTFCEsigch]
    assert any(c.p <= 0.05 for c in res.result.clusters)

    # negative control: group A split in two, its own counts; no effect exists
    a, ca = (pd.read_csv(DATA / f) for f in ("circ_phase_groupA.csv", "circ_phase_countsA.csv"))
    for name, rows in (("1", slice(0, 8)), ("2", slice(8, 16))):
        a.iloc[rows].to_csv(tmp_path / f"a{name}.csv", index=False)
        ca.iloc[rows].to_csv(tmp_path / f"c{name}.csv", index=False)
    null = quiet(circ_pipeline.run_circ, group_params(
        data1_file=str(tmp_path / "a1.csv"), data2_file=str(tmp_path / "a2.csv"),
        count1_file=str(tmp_path / "c1.csv"), count2_file=str(tmp_path / "c2.csv"),
        permutations=199))
    assert null.correctTFCEsigch.size == 0
    assert all(c.p > 0.05 for c in null.result.clusters)


def test_anglinear_planted_association():
    res = quiet(circ_pipeline.run_circ, anglinear_params(permutations=199))
    assert res.prepared.gi is None and res.engine.rho is not None
    assert res.correctTFCEsigch.size > 0
    assert len(res.descriptive["group"]) == 1


# ---------------------------------------------------------------------------- Tier-1 guard

@pytest.fixture
def stamped(tmp_path):
    rng = np.random.default_rng(4)
    A = np.mod(0.3 + 0.4 * rng.standard_normal((12, 6)), 2 * np.pi)
    labels = [f"E{k}" for k in range(1, 7)]

    def write(measure):
        Y, info = circ.linearise(A, measure, count_file="spindle_counts.csv")
        T = circ.linearise_table(Y, info, labels, np.arange(1, 13))
        T.insert(1, "group", np.repeat(["a", "b"], 6))
        T.insert(2, "n_events", rng.integers(20, 60, 12))
        path = tmp_path / f"{measure}.csv"
        T.to_csv(path, index=False)
        return str(path)

    return write


def test_tier1_guard(stamped):
    unsigned = stamped("unsigned")
    for preset in ("anova1", "pairedT", "mixed2way"):
        with pytest.raises(circ.CircInputError, match="has no covariate slot") as e:
            circ_pipeline.tier1_guard({"comparison": preset, "data_file": unsigned})
        assert e.value.code == "circUnsignedNeedsCountCovariate"
    with pytest.raises(circ.CircInputError, match="does not contain the event-count column"):
        circ_pipeline.tier1_guard({"comparison": "ancova", "data_file": unsigned,
                                   "covariate_cols": ["age"]})
    ok = circ_pipeline.tier1_guard({"comparison": "ancova", "data_file": unsigned,
                                    "covariate_cols": ["age", "spindle_count"]})
    assert ok["checked"] and ok["stamp"]["measure"] == "circ_unsigned_distance"
    assert circ_pipeline.tier1_guard({"comparison": "regression", "data_file": unsigned,
                                      "covariate_cols": "n_events",
                                      "circ_count_col": "n_events"})["checked"]
    assert circ_pipeline.tier1_guard({"comparison": "regression", "data_file": unsigned,
                                      "covariate_cols": "spindle_counts"})["checked"]
    signed = circ_pipeline.tier1_guard({"comparison": "anova1", "data_file": stamped("signed")})
    assert signed["checked"]
    assert not circ_pipeline.tier1_guard(group_params())["checked"]


def test_tier1_guard_runs_before_glm_and_legacy(stamped):
    unsigned = stamped("unsigned")
    with pytest.raises(circ.CircInputError, match="UNSIGNED"):
        pipeline.run({"comparison": "anova1", "data_file": unsigned, "group_col": "group",
                      "meta_cols": ["Subject", "group", "n_events", "circ_provenance"]})
    with pytest.raises(circ.CircInputError, match="UNSIGNED"):
        pipeline.prepare({"comparison": "unpairedT", "data1_file": unsigned,
                          "data2_file": unsigned})


# ---------------------------------------------------------------------------- outputs

@pytest.mark.parametrize("make", [group_params, anglinear_params,
                                  lambda **kw: group_params(comparison="circ_phase_group_u2",
                                                            **kw)])
def test_writes_results_workbook_and_report(make, tmp_path):
    paths = quiet(run_and_write, make(output_path=str(tmp_path), permutations=50))
    assert all(p.exists() for p in paths.values())
    s = json.loads(paths["json"].read_text())
    e = s["effects"][0]
    assert e["cluster_statistic"] == "mass" and s["alpha"] == 0.05
    c = e["circular"]
    assert c["descriptive"]["note"].startswith("Descriptive only")
    assert c["exclusion_reason"] == circ_pipeline.exclusion_reason(c["comparison"])
    assert s["tfce"]["dh"] == (0.005 if c["statistic"] == "watson_U2" else 0.1)
    z = np.load(paths["npz"])
    assert {"main/stat", "main/tfce", "main/p_tfce", "main/evaluable",
            "main/mean_direction_deg"} <= set(z.files)
    html = paths["html"].read_text()
    assert "Descriptive circular summary (no inference)" in html
    assert "Whole-head test" not in html and "rotation of +0.0 deg" in html


# ---------------------------------------------------------------------------- T12 missing y

def test_anglinear_missing_measure_dropped_up_front(tmp_path):
    """Mirror of MATLAB ``test_circ_snpm`` T12: subjects with a non-finite measure leave before
    anything else, the run equals one on the pre-filtered file, and the per-channel n is the same
    in the observed and every permuted map."""
    rng = np.random.default_rng(12)
    ang = pd.read_csv(DATA / "circ_anglinear_angles.csv")
    meas = pd.read_csv(DATA / "circ_anglinear_measure.csv")
    chans = [c for c in ang.columns if c != "Subject"]
    for c in rng.choice(chans, 20, replace=False):                  # missing angles stay
        ang.loc[rng.choice(len(ang), 3, replace=False), c] = np.nan
    miss = np.sort(rng.choice(len(meas), 6, replace=False))
    meas["measure"] = meas["measure"].astype(object)
    meas.loc[miss, "measure"] = np.nan
    meas.loc[miss[-1], "measure"] = "Inf"                           # literal Inf counts as missing
    ang.to_csv(tmp_path / "a.csv", index=False)
    meas.to_csv(tmp_path / "m.csv", index=False)
    keep = np.setdiff1d(np.arange(len(meas)), miss)
    meas.iloc[keep].to_csv(tmp_path / "m_pre.csv", index=False)
    dropped_ids = meas["Subject"].iloc[miss].tolist()

    p = anglinear_params(data1_file=str(tmp_path / "a.csv"), data2_file=str(tmp_path / "m.csv"))
    with pytest.warns(UserWarning, match=r"^6 of 30 matched subjects") as rec:
        circ_pipeline.prepare_circ(p)
    msg = next(str(w.message) for w in rec if "linear measure" in str(w.message))
    assert all(s in msg for s in dropped_ids)

    perms = np.argsort(np.random.default_rng(1212).random((300, len(keep))), axis=1)
    res = quiet(circ_pipeline.run_circ, p, perms=perms)
    pre = quiet(circ_pipeline.run_circ, dict(p, data2_file=str(tmp_path / "m_pre.csv")),
                perms=perms)
    pr = res.prepared
    assert pr.dropped_subjects == dropped_ids and pr.measure_nan_removed == 6 and pr.n1g == 24
    assert pre.prepared.measure_nan_removed == 0
    assert any(n.startswith("Subjects dropped for a missing measure value: 6") for n in pr.notes)
    a, b = res.result, pre.result
    for name in ("real_T", "p_real", "real_TFCE", "tMaxTFCE", "maxMass", "p_correctedTFCE"):
        assert np.array_equal(getattr(a, name), getattr(b, name), equal_nan=True), name
    assert np.array_equal(res.engine.rho, pre.engine.rho, equal_nan=True)
    assert np.array_equal(pr.evaluable, pre.prepared.evaluable)
    assert np.array_equal(pr.per_channel_n, pre.prepared.per_channel_n)
    assert [(c.channels.tolist(), c.p) for c in a.clusters] == \
        [(c.channels.tolist(), c.p) for c in b.clusters]

    # per-channel n fixed across the null
    assert np.isfinite(pr.y).all() and pr.y.size == 24
    n0 = circ.corr_anglinear(pr.y, pr.A, valid_channels=pr.evaluable).n
    assert np.array_equal(n0, pr.per_channel_n) and (n0 < 24).any()
    fit, resid = res.engine.fl_fit[:, 0], res.engine.fl_resid[:, 0]
    for ix in perms:
        assert np.array_equal(circ.corr_anglinear(fit + resid[ix], pr.A,
                                                  valid_channels=pr.evaluable).n, n0)
    # control: the old scheme (NaN y left in the residual) moves the per-channel n
    A_old = np.full((30, pr.A.shape[1]), np.nan)
    A_old[keep] = pr.A
    y_old = np.full(30, np.nan)
    y_old[keep] = pr.y
    r_old = y_old - np.nanmean(y_old)
    n_old = circ.corr_anglinear(r_old, A_old, valid_channels=pr.evaluable).n
    assert any(not np.array_equal(circ.corr_anglinear(r_old[rng.permutation(30)], A_old,
                                                      valid_channels=pr.evaluable).n, n_old)
               for _ in range(50))

    # only 7 finite measures: refused
    m7 = meas.copy()
    m7["measure"] = np.nan
    m7.loc[:6, "measure"] = pd.read_csv(DATA / "circ_anglinear_measure.csv")["measure"][:7]
    m7.to_csv(tmp_path / "m7.csv", index=False)
    with pytest.raises(circ.CircInputError) as e:
        quiet(circ_pipeline.prepare_circ, dict(p, data2_file=str(tmp_path / "m7.csv")))
    assert e.value.code == "circTooFewSubjects"


def test_dropped_subjects_reach_the_files(tmp_path):
    meas = pd.read_csv(DATA / "circ_anglinear_measure.csv")
    meas.loc[[2, 5], "measure"] = np.nan
    meas.to_csv(tmp_path / "m.csv", index=False)
    paths = quiet(run_and_write, anglinear_params(data2_file=str(tmp_path / "m.csv"),
                                                  output_path=str(tmp_path), permutations=50))
    c = json.loads(paths["json"].read_text())["effects"][0]["circular"]
    assert c["measure_nan_removed"] == 2 and c["dropped_subjects"] == ["sub003", "sub006"]
    assert "Subjects dropped for a missing measure value: 2 (sub003, sub006)" in \
        paths["html"].read_text()


def test_methods_text_matches_the_design():
    from hdeeg_snpm.views import effect_views

    lin = effect_views(quiet(circ_pipeline.run_circ, anglinear_params(permutations=20)))[0]
    assert "at least 8 subjects with both an angle and the measure" in lin.methods
    assert "per group" not in lin.methods and "count" not in lin.methods.lower()
    grp = effect_views(quiet(circ_pipeline.run_circ, group_params(permutations=20)))[0]
    assert "at least 8 subjects per group" in grp.methods
