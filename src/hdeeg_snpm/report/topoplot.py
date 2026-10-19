"""Scalp maps in EEGLAB's polar projection (nose up, head outline at radius 0.60)."""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator

HEAD_RADIUS = 0.60


def positions(theta_deg, radius):
    """EEGLAB polar coordinates to plot coordinates: angle 0 is the nose, positive to the right."""
    t = np.deg2rad(np.asarray(theta_deg, dtype=float))
    r = np.asarray(radius, dtype=float)
    return r * np.sin(t), r * np.cos(t)


def topoplot(values, setup, *, ax=None, title="", marks=None, numbers=None, symmetric=True,
             cmap=None, label="", grid=140):
    """Draw one map. ``marks`` are channel indices drawn as filled dots (significant channels);
    ``numbers`` maps a cluster number to its channel indices and writes the number at the
    cluster's centre. Channels with a missing value are left out of the interpolation and
    drawn as open grey circles."""
    values = np.asarray(values, dtype=float)
    # Like EEGLAB's squeeze: pull every electrode inward so the outermost one sits on the head
    # outline, then colour only inside the outline.
    squeeze = min(1.0, HEAD_RADIUS / float(np.max(setup.radius)))
    x, y = positions(setup.theta, np.asarray(setup.radius, dtype=float) * squeeze)
    plot_r = HEAD_RADIUS
    ax = ax or plt.subplots(figsize=(3.6, 3.6))[1]
    ok = np.isfinite(values)
    mesh = None
    if ok.sum() >= 4:
        gx, gy = np.meshgrid(np.linspace(-plot_r, plot_r, grid), np.linspace(-plot_r, plot_r, grid))
        # Thin-plate spline, close to EEGLAB's 'v4': smooth to the rim, where a cubic
        # triangulation stops at the electrode hull and nearest-neighbour fill makes wedges.
        rbf = RBFInterpolator(np.column_stack([x[ok], y[ok]]), values[ok],
                              kernel="thin_plate_spline")
        z = rbf(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
        z[np.hypot(gx, gy) > plot_r] = np.nan
        vmax = np.nanmax(np.abs(values[ok])) or 1.0
        lim = (-vmax, vmax) if symmetric else (float(np.nanmin(values[ok])), float(vmax))
        mesh = ax.pcolormesh(gx, gy, z, cmap=cmap or ("RdBu_r" if symmetric else "viridis"),
                             vmin=lim[0], vmax=lim[1], shading="auto", rasterized=True)
    a = np.linspace(0, 2 * np.pi, 200)
    ax.plot(HEAD_RADIUS * np.cos(a), HEAD_RADIUS * np.sin(a), color="k", lw=1.2)
    ax.plot([-0.07, 0, 0.07], [HEAD_RADIUS * 0.993, HEAD_RADIUS * 1.12, HEAD_RADIUS * 0.993],
            color="k", lw=1.2)
    ax.scatter(x[ok], y[ok], s=2, color="k", alpha=0.35, linewidths=0)
    if (~ok).any():
        ax.scatter(x[~ok], y[~ok], s=9, facecolors="none", edgecolors="0.55", linewidths=0.6)
    if marks is not None and len(marks):
        ax.scatter(x[marks], y[marks], s=16, color="k", edgecolors="w", linewidths=0.5, zorder=3)
    for number, ch in (numbers or {}).items():
        ax.text(x[ch].mean(), y[ch].mean(), str(number), ha="center", va="center", fontsize=9,
                fontweight="bold", color="k", zorder=4,
                bbox={"boxstyle": "circle,pad=0.15", "fc": "w", "ec": "k", "lw": 0.6})
    ax.set_aspect("equal")
    ax.set_xlim(-plot_r * 1.05, plot_r * 1.05)
    ax.set_ylim(-plot_r * 1.05, plot_r * 1.16)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    if mesh is not None:
        ax.figure.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02, label=label)
    return ax


def to_png(fig, dpi=150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def data_uri(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def effect_figure(view, setup, alpha) -> bytes:
    """Three maps for one effect: the statistic, TFCE-significant channels, and the numbered
    significant clusters."""
    sig = view.significant(alpha)
    signed = view.stat_name not in ("F", "U2")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    topoplot(view.stat, setup, ax=axes[0], title=f"{view.stat_name} map", symmetric=signed,
             label=view.stat_name, marks=sig["uncorrected"])
    topoplot(view.stat, setup, ax=axes[1], title=f"TFCE-corrected (p ≤ {alpha:g})",
             symmetric=signed, label=view.stat_name, marks=sig["tfce"])
    topoplot(view.stat, setup, ax=axes[2], title=f"Cluster-corrected (p ≤ {alpha:g})",
             symmetric=signed, label=view.stat_name, marks=sig["cluster"],
             numbers={c.number: c.channels for c in view.clusters if c.p <= alpha})
    return to_png(fig)
