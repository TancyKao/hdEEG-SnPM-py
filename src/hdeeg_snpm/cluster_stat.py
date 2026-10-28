"""How a cluster is scored in the tiers that use a cluster mass (GLM, circular, mixed model).

A cluster is a connected set of channels whose model p is below the cluster-forming alpha. Its
score is the sum of the Wald statistic (t² or F) over its channels ("sum", the default: a
summed-statistic cluster mass in the sense of Maris & Oostenveld 2007, taken here on the Wald
scale) or that sum divided by the number of channels
("mean", the rule before version 0.8.0, which reproduces older runs). The threshold, the Wald
scale and the permutations are the same under both rules; TFCE does not use this score.

Ports ``snpm_cluster_statistic.m``. Every text that names the rule is in this module, so the
wording can follow MATLAB's in one place.
"""

from __future__ import annotations

import numpy as np

RULES = ("sum", "mean")
DEFAULT = "sum"

# MATLAB error id ``snpm:badClusterStatistic``; texts shared with MATLAB word for word
BAD_MESSAGE = "cluster_statistic must be 'sum' or 'mean'; got '{value}'."
NON_TEXT = "<non-text value>"     # how a value that is not text is shown in BAD_MESSAGE


class BadClusterStatistic(ValueError):
    """MATLAB ``snpm:badClusterStatistic``: ``cluster_statistic`` is neither 'sum' nor 'mean'."""

    code = "badClusterStatistic"


def resolve(value=None) -> str:
    """The rule named by ``value``, as ``snpm_cluster_statistic.m`` reads it: absent, empty or
    blank gives 'sum'; text is trimmed and lower-cased; anything other than 'sum' or 'mean' is
    refused, quoting the value as given (``<non-text value>`` when it is not text)."""
    if value is None:
        return DEFAULT
    if not isinstance(value, str):
        raise BadClusterStatistic(BAD_MESSAGE.format(value=NON_TEXT))
    rule = value.strip().lower() or DEFAULT
    if rule not in RULES:
        raise BadClusterStatistic(BAD_MESSAGE.format(value=value))
    return rule


def score(wald: np.ndarray, channels: np.ndarray, rule: str) -> float:
    """The score of one cluster: sum of ``wald`` over ``channels`` (NaN ignored), divided by
    the number of channels under 'mean'."""
    s = np.nansum(wald[channels])
    return s / channels.size if rule == "mean" else s


# ---------------------------------------------------------------- wording
# Shared with MATLAB (snpm_cluster_statistic.m, core_snpm_circ.m, generateAnalysisReport.m);
# do not reword. Under "sum" no text says "divided by the number of channels".

GLM_WHAT = "t² (or F)"            # GLM and mixed model: the summed Wald statistic
CIRC_WHAT = "the statistic"       # circular: Hotelling F, Watson U² or circular-linear F, as is
_MEAN_TAIL = ", divided by the number of channels (mean; reproduces runs before this change)"
_PREFIX = "Cluster statistic: "
_HELFRICH = " This differs from the FieldTrip maxsize (cluster extent) used by Helfrich et al."


def rule_text(rule: str, what: str = GLM_WHAT) -> str:
    """'sum of <what> over the cluster', plus the mean tail under 'mean'."""
    return f"sum of {what} over the cluster" + (_MEAN_TAIL if rule == "mean" else "")


def results_line(rule: str, what: str = GLM_WHAT) -> str:
    """The results-text line of the GLM and mixed-model tiers (MATLAB's ``line``)."""
    return _PREFIX + rule_text(rule, what)


def circ_results_line(rule: str) -> str:
    """The circular tier's results-text line."""
    return results_line(rule, CIRC_WHAT) + "." + _HELFRICH


def circ_recorded(rule: str) -> str:
    """``results_struct.circ.cluster_statistic``: the circular line without its prefix
    (``regexprep(cs_line, '^Cluster statistic: ', '')`` in ``core_snpm_circ.m``)."""
    return rule_text(rule, CIRC_WHAT)


def card_value(rule: str, what: str = GLM_WHAT) -> str:
    """The report's model-card and settings value: the rule in words, capitalised."""
    t = rule_text(rule, what)
    return t[0].upper() + t[1:]


# the report's notice under a circular cluster view (HTML)
CIRC_NOTICE = {
    "sum": "Cluster statistic: <b>sum</b> of the statistic over the cluster, not cluster extent.",
    "mean": ("Cluster statistic: <b>mean</b> (sum of the statistic over the cluster, divided by "
             "the number of channels), not cluster extent."),
}

# Excel ``settings`` sheet: one row, these columns (MATLAB writes the same sheet)
SETTINGS_COLUMNS = ("cluster_statistic", "alpha")
