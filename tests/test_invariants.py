"""Rules that keep the permutation test valid when data are missing.

These restate the MATLAB suites ``test_perm_invariant`` and ``test_global_common_channels``:
the observed map and every permuted map must cover the same channels on the same subjects, and
the whole-head test must average every unit over one channel set.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from hdeeg_snpm import legacy, perm
from hdeeg_snpm.montage import get_montage
from hdeeg_snpm.pipeline import prepare

LABELS = list(get_montage("egi").labels)
N_CH = len(LABELS)


def write(folder, name, matrix, ids=None):
    frame = pd.DataFrame(matrix, columns=LABELS[:matrix.shape[1]])
    frame.insert(0, "Subject", ids or [f"sub{i:03d}" for i in range(1, len(matrix) + 1)])
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


def params(folder, comparison, x, y, **extra):
    return {"comparison": comparison, "channels": "egi", "datatype": "absolute",
            "data1_file": write(folder, "x", x), "data2_file": write(folder, "y", y), **extra}


def quiet_prepare(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prepare(p)


@pytest.fixture
def holes():
    """30 subjects; scattered missing cells in both matrices, different places in each."""
    rng = np.random.default_rng(5)
    x, y = rng.normal(size=(30, N_CH)), rng.normal(size=(30, N_CH))
    x[rng.integers(0, 30, 12), rng.integers(0, N_CH, 12)] = np.nan
    y[rng.integers(0, 30, 12), rng.integers(0, N_CH, 12)] = np.nan
    return x, y


@pytest.mark.parametrize("comparison", ["correlationP", "correlationS"])
@pytest.mark.parametrize("with_covariates", [False, True])
def test_correlation_null_covers_the_observed_channels_on_the_same_subjects(
        tmp_path, holes, comparison, with_covariates):
    x, y = holes
    extra = {}
    if with_covariates:
        cov = pd.DataFrame({"Subject": [f"sub{i:03d}" for i in range(1, 31)],
                            "age": np.random.default_rng(1).normal(size=30)})
        cov.to_csv(tmp_path / "cov.csv", index=False)
        extra = {"use_covariates": True, "covariate_file": str(tmp_path / "cov.csv")}
    prep = quiet_prepare(params(tmp_path, comparison, x, y, **extra))
    incomplete = np.isnan(x).any(axis=0) | np.isnan(y).any(axis=0)
    assert np.array_equal(~prep.rule.evaluable, incomplete) and incomplete.sum() > 5
    # every retained channel is complete, so its n cannot depend on the permutation
    assert np.isfinite(prep.x[:, prep.rule.evaluable]).all()
    assert np.isfinite(prep.y[:, prep.rule.evaluable]).all()
    assert set(prep.n_analysed[prep.rule.evaluable]) == {30}
    design = legacy._design(prep.x, prep.y, prep.stat_comparison, "both")
    observed = design.observed(True)[0]
    assert np.array_equal(np.isnan(observed), incomplete)
    for row in perm.row_permutations(30, 300, np.random.default_rng(2)):
        r = design.stat(row)
        assert np.array_equal(np.isnan(r), incomplete)
        assert np.nanmax(np.abs(r)) <= 1 + 1e-12


def test_rank_partial_correlation_equals_the_textbook_formula(tmp_path):
    rng = np.random.default_rng(8)
    z = rng.normal(size=30)
    x = rng.normal(size=(30, N_CH)) + z[:, None]
    y = rng.normal(size=(30, N_CH)) + z[:, None]
    pd.DataFrame({"Subject": [f"sub{i:03d}" for i in range(1, 31)], "z": z}).to_csv(
        tmp_path / "cov.csv", index=False)
    prep = quiet_prepare(params(tmp_path, "correlationS", x, y, use_covariates=True,
                                covariate_file=str(tmp_path / "cov.csv")))
    assert prep.stat_comparison == "correlationP" and prep.ncov == 1
    got = legacy.corr_columns(prep.x, prep.y)
    for ch in (0, 17, 100):
        rxy = stats.spearmanr(x[:, ch], y[:, ch]).statistic
        rxz = stats.spearmanr(x[:, ch], z).statistic
        ryz = stats.spearmanr(y[:, ch], z).statistic
        expected = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
        assert abs(got[ch] - expected) < 1e-12


def test_a_subject_with_a_missing_covariate_is_dropped_whole(tmp_path):
    rng = np.random.default_rng(9)
    x, y = rng.normal(size=(12, N_CH)), rng.normal(size=(12, N_CH))
    age = rng.normal(size=12)
    age[4] = np.nan
    pd.DataFrame({"Subject": [f"sub{i:03d}" for i in range(1, 13)], "age": age}).to_csv(
        tmp_path / "cov.csv", index=False)
    p = params(tmp_path, "correlationP", x, y, use_covariates=True,
               covariate_file=str(tmp_path / "cov.csv"))
    with pytest.warns(UserWarning, match="missing covariate"):
        prep = prepare(p)
    assert prep.x.shape[0] == 11 and "sub005" not in prep.subjects
    assert prep.rule.evaluable.all()          # the map is not turned into NaN


def test_paired_test_keeps_incomplete_channels_but_drops_degenerate_ones(tmp_path):
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(10, N_CH)), rng.normal(size=(10, N_CH))
    x[:3, 5] = np.nan                 # 7 usable pairs: tested on those
    x[1:, 9] = np.nan                 # 1 usable pair: zero degrees of freedom
    prep = quiet_prepare(params(tmp_path, "pairedT", x, y))
    assert prep.rule.excluded.tolist() == [9]
    assert prep.n_analysed[5] == 7 and prep.n_analysed[9] == 0
    design = legacy._design(prep.x, prep.y, "pairedT", "both")
    for row in perm.signflip_patterns(10, 300, np.random.default_rng(1)):
        t = design.stat(row)
        assert np.isnan(t[9]) and np.isfinite(np.delete(t, 9)).all()


def test_unpaired_test_drops_channels_that_can_leave_one_subject_in_a_group(tmp_path):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(8, N_CH)), rng.normal(size=(10, N_CH))
    y[:, 3] = np.nan                  # missing in one group only: 8 usable, needs 12
    x[:2, 7] = np.nan                 # 16 usable: stays
    prep = quiet_prepare(params(tmp_path, "unpairedT", x, y))
    assert prep.rule.min_units == 12 and prep.rule.excluded.tolist() == [3]
    design = legacy._design(prep.x, prep.y, "unpairedT", "both")
    for row in perm.relabel_assignments(18, 8, 300, np.random.default_rng(1)):
        t = design.stat(row)
        assert np.isnan(t[3]) and np.isfinite(np.delete(t, 3)).all()


# --------------------------------------------------------------------------- whole-head test

def gradient_pair(n=24, seed=0):
    """Two conditions with no difference and a strong front-to-back power gradient."""
    rng = np.random.default_rng(seed)
    gradient = np.linspace(20, 0, N_CH)
    subject = rng.normal(size=(n, 1))
    x = gradient + subject + 0.3 * rng.normal(size=(n, N_CH))
    y = gradient + subject + 0.3 * rng.normal(size=(n, N_CH))
    return x, y


def test_whole_head_test_on_complete_data_is_the_plain_average_test():
    x, y = gradient_pair()
    g = legacy.global_test(x, y, "pairedT")
    ref = stats.ttest_rel(x.mean(axis=1), y.mean(axis=1))
    assert g.n_channels == N_CH and abs(g.stat - ref.statistic) < 1e-12
    assert abs(g.pval - ref.pvalue) < 1e-12


def test_whole_head_test_averages_both_conditions_over_one_channel_set():
    """Different bad channels per condition: averaging each over its own channels rejects on
    null data, because the two averages then cover different parts of the gradient."""
    x, y = gradient_pair()
    x[:, :30] = np.nan                # condition A lost frontal channels
    y[:, -30:] = np.nan               # condition B lost posterior channels
    naive = stats.ttest_rel(np.nanmean(x, axis=1), np.nanmean(y, axis=1))
    assert naive.pvalue < 1e-10
    g = legacy.global_test(x, y, "pairedT")
    assert g.n_channels == N_CH - 60 and g.pval > 0.05
    assert not g.channels_kept[:30].any() and not g.channels_kept[-30:].any()


def test_whole_head_false_positive_rate_is_nominal_with_mismatched_bad_channels():
    rejections = 0
    for seed in range(400):
        x, y = gradient_pair(seed=seed)
        x[:, :30] = np.nan
        y[:, -30:] = np.nan
        rejections += legacy.global_test(x, y, "pairedT").pval <= 0.05
    assert 0.025 <= rejections / 400 <= 0.08


def test_whole_head_test_treats_infinite_values_as_unusable():
    x, y = gradient_pair()
    x[2, 10] = np.inf
    g = legacy.global_test(x, y, "unpairedT")
    assert not g.channels_kept[10] and np.isfinite(g.stat)


def test_whole_head_test_with_no_common_channel_is_undefined_and_says_so():
    x, y = gradient_pair(n=6)
    x[0, :] = np.nan
    with pytest.warns(UserWarning, match="no channel is usable"):
        g = legacy.global_test(x, y, "pairedT")
    assert np.isnan(g.stat) and np.isnan(g.pval) and g.n_channels == 0


def test_whole_head_correlation_is_unchanged_by_masking_twice(tmp_path, holes):
    x, y = holes
    prep = quiet_prepare(params(tmp_path, "correlationP", x, y))
    a = legacy.global_test(prep.x, prep.y, "correlationP")
    keep = a.channels_kept
    b = legacy.global_test(np.where(keep, prep.x, np.nan), np.where(keep, prep.y, np.nan),
                           "correlationP")
    assert a.stat == b.stat and a.pval == b.pval and a.n_channels == int(prep.rule.evaluable.sum())


def test_channels_with_no_defined_statistic_are_reported_not_left_as_silent_gaps(tmp_path):
    rng = np.random.default_rng(6)
    x = np.abs(rng.normal(5, 1, size=(12, N_CH)))
    y = np.abs(rng.normal(5, 1, size=(12, N_CH)))
    x[0, 10] = 0.0                     # log10(0) = -Inf: an unusable cell, not a lost channel
    y[:, 20] = x[:, 20]                # identical in both conditions: no variance
    p = params(tmp_path, "pairedT", x, y, datatype="logscale")
    with pytest.warns(UserWarning, match="identical values in both conditions"):
        prep = prepare(p)
    assert prep.n_analysed[10] == 11 and prep.rule.evaluable[10]
    assert prep.rule.excluded.tolist() == [20]
    assert prep.rule.reasons == ["identical values in both conditions"]
    t = legacy._design(prep.x, prep.y, "pairedT", "both").observed(True)[0]
    assert np.isfinite(np.delete(t, 20)).all() and np.isnan(t[20])


def test_infinite_observed_value_is_counted_strictly():
    null = np.array([np.inf, 3.0, 2.0])
    assert legacy._fwe_p(np.array([np.inf]), null, True)[0] == 0.5
