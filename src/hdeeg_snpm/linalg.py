"""Rank of a design and which of its columns are aliased, shared by the GLM and mixed-model
checks (``snpm_glm_estimability.m``, ``snpm_lmm_check_effect.m``)."""

from __future__ import annotations

import numpy as np


def rank_tol(X: np.ndarray) -> float:
    """``max(X.shape) * eps * s_max(X)``: numpy's default rank tolerance, written out as the
    MATLAB checks write it (``rank(X, tol)``)."""
    s = np.linalg.svd(X, compute_uv=False)
    return max(X.shape) * 2.220446049250313e-16 * (s[0] if s.size else 0.0)


def aliased_columns(X: np.ndarray, names, order, tol: float, zero_text: str):
    """Left-to-right walk over the columns ``order`` of X: a column is kept when it raises the
    rank of the kept columns, else it is aliased. Returns ``(kept, [(j, item)])``; the item names
    the kept columns k with ``|w_k| * norm(x_k) / norm(x_j) > 1e-6`` in the least-squares fit of
    x_j on them, or ``'<name> <zero_text>'`` when there are none."""
    kept: list[int] = []
    items = []
    for j in order:
        if np.linalg.matrix_rank(X[:, kept + [j]], tol) > len(kept):
            kept.append(j)
            continue
        xj = X[:, j]
        of = []
        nj = np.linalg.norm(xj)
        if kept and nj > 0:
            Xk = X[:, kept]
            w = np.linalg.lstsq(Xk, xj, rcond=None)[0]
            rel = np.abs(w) * np.linalg.norm(Xk, axis=0) / nj
            of = [names[k] for k, r in zip(kept, rel) if r > 1e-6]
        items.append((j, f"{names[j]} {zero_text}" if not of
                      else f"{names[j]} is a linear combination of {', '.join(of)}"))
    return kept, items
