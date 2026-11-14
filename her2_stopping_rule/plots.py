"""Figures for the pitch.

Every figure is produced by the step that owns its data, so the numbers on a
slide are the numbers that actually ran. No figure invents a value: where a step
did not run, its figure is skipped rather than drawn empty.

Colour comes from the validated reference palette, slots 1-3 only. That subset
is documented as clearing the all-pairs colour-vision and normal-vision floors
in both light and dark modes, which matters because these figures end up on a
projector. The validator itself needs node, which is not on this VM, so the
palette is used as published rather than re-derived.

Conventions held across every figure: one axis per chart and never two y-scales,
thin marks, recessive grid and spines, direct labels in text colours rather than
series colours, and a legend whenever two or more series share a plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

#: Categorical slots 1-3 of the reference palette, light mode.
SERIES: Final[tuple[str, str, str]] = ("#2a78d6", "#eb6834", "#1baf7a")

#: Ink and surface tokens. Text never wears a series colour.
TEXT_PRIMARY: Final[str] = "#0b0b0b"
TEXT_SECONDARY: Final[str] = "#52514e"
SURFACE: Final[str] = "#fcfcfb"
GRID: Final[str] = "#dcdbd6"

#: Status colours, reserved and never reused as a fourth series.
GOOD: Final[str] = "#008300"
CRITICAL: Final[str] = "#e34948"


def _style(ax: plt.Axes) -> plt.Axes:
    """Apply the shared recessive styling to an axis.

    Args:
        ax: The axis to style.

    Returns:
        The same axis, for chaining.
    """
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)
    return ax


def _save(fig: plt.Figure, path: Path) -> Path:
    """Write a figure and close it.

    Args:
        fig: The figure to save.
        path: Destination PNG.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def funnel(records: Sequence[tuple[str, int, int, float]], path: Path) -> Path:
    """Draw how many designs survived each step.

    A single series, so no legend: the title names it. Counts are direct-labelled
    because the whole point of the figure is the specific numbers.

    Args:
        records: ``(step, rows_in, rows_out, seconds)`` per step, in order.
        path: Destination PNG.

    Returns:
        The path written.
    """
    rows = [(s, out) for s, _, out, _ in records if out > 0]
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.42 * len(rows) + 1.2)))
    _style(ax)

    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    positions = np.arange(len(rows))
    ax.barh(positions, values, height=0.6, color=SERIES[0], zorder=3)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("designs surviving (log scale)")
    ax.set_title("Funnel: designs surviving each step", fontsize=12, loc="left")
    for y, v in zip(positions.tolist(), values):
        ax.text(v * 1.15, y, f"{v:,}", va="center", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(right=max(values) * 3)
    return _save(fig, path)


def surrogate_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    test_distance: int,
    base_rate: float,
    path: Path,
) -> Path:
    """Draw the surrogate's held-out ROC curve.

    Holding out an entire edit-distance stratum is the honest test: a random
    split would leak, because variants one mutation apart are near-duplicates.

    Args:
        fpr: False-positive rates.
        tpr: True-positive rates.
        auc: Area under the curve.
        test_distance: The edit distance held out.
        base_rate: Fraction of held-out variants that are high binders.
        path: Destination PNG.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5))
    _style(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--", zorder=2)
    ax.plot(fpr, tpr, color=SERIES[0], linewidth=2, zorder=3)
    ax.text(
        0.55, 0.22, f"ROC-AUC {auc:.3f}", fontsize=13, color=TEXT_PRIMARY, weight="bold"
    )
    ax.text(
        0.55,
        0.14,
        f"held out distance {test_distance}\nbase rate {base_rate:.1%}",
        fontsize=9,
        color=TEXT_SECONDARY,
    )
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(
        f"Affinity surrogate, distance {test_distance} held out",
        fontsize=12,
        loc="left",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _save(fig, path)


def distance_tradeoff(
    distances: Sequence[int],
    p_high: Sequence[float],
    eligible: Sequence[float | None],
    path: Path,
) -> Path:
    """Draw the affinity and novelty curves that set the design distance.

    Two series on one y-axis, both percentages, so no dual axis is needed or
    allowed. Both are direct-labelled as well as legended, so identity never
    rests on colour alone.

    Args:
        distances: CDR3 mutation counts.
        p_high: Measured fraction of high binders at each distance.
        eligible: Fraction clearing prior art, or None where unmeasured.
        path: Destination PNG.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _style(ax)

    ax.plot(
        distances,
        p_high,
        color=SERIES[0],
        linewidth=2,
        marker="o",
        markersize=5,
        zorder=4,
        label="measured high binders",
    )
    known = [(d, e) for d, e in zip(distances, eligible) if e is not None]
    if known:
        ax.plot(
            [d for d, _ in known],
            [e for _, e in known],
            color=SERIES[1],
            linewidth=2,
            marker="o",
            markersize=5,
            linestyle="--",
            zorder=4,
            label="clears prior art",
        )

    ax.axvspan(min(distances) - 0.5, 2.5, color=CRITICAL, alpha=0.07, zorder=1)
    ax.text(
        1.5,
        92,
        "below the\nmutation floor",
        fontsize=8.5,
        color=CRITICAL,
        ha="center",
        va="top",
    )

    ax.set_xlabel("CDR3 mutations from trastuzumab")
    ax.set_ylabel("percent")
    ax.set_title(
        "Affinity falls faster than novelty eligibility rises", fontsize=12, loc="left"
    )
    ax.set_ylim(0, 100)
    ax.set_xticks(list(distances))
    legend = ax.legend(frameon=False, fontsize=9, loc="center right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    return _save(fig, path)


def calibration(
    negatives: np.ndarray, cutoff: float, achieved_fpr: float, path: Path
) -> Path:
    """Draw where the ipTM cutoff falls among measured non-binders.

    Args:
        negatives: ipTM scores of pairs measured as non-binding.
        cutoff: The chosen threshold.
        achieved_fpr: Fraction of negatives the cutoff admits.
        path: Destination PNG.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax)
    ax.hist(negatives, bins=40, color=SERIES[0], alpha=0.85, zorder=3)
    ax.axvline(cutoff, color=CRITICAL, linewidth=2, zorder=4)
    ax.text(
        cutoff,
        ax.get_ylim()[1] * 0.92,
        f"  cutoff {cutoff:.3f}\n  admits {achieved_fpr:.1%}",
        fontsize=9.5,
        color=CRITICAL,
        va="top",
    )
    ax.set_xlabel("ipTM of pairs measured as NON-binding")
    ax.set_ylabel("count")
    ax.set_title(
        "The cutoff is calibrated on measured non-binders", fontsize=12, loc="left"
    )
    return _save(fig, path)


def model_spread(spreads: np.ndarray, threshold: float, path: Path) -> Path:
    """Draw the two-model disagreement that sets the stopping threshold.

    Args:
        spreads: Absolute ipTM differences between the two folding models.
        threshold: The stopping rule's threshold.
        path: Destination PNG.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _style(ax)
    ax.hist(spreads, bins=30, color=SERIES[2], alpha=0.85, zorder=3)
    if np.isfinite(threshold):
        ax.axvline(threshold, color=TEXT_PRIMARY, linewidth=2, zorder=4)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.9,
            f"  stopping threshold {threshold:.4f}",
            fontsize=9.5,
            color=TEXT_PRIMARY,
            va="top",
        )
    ax.set_xlabel("absolute ipTM difference between the two models")
    ax.set_ylabel("designs")
    ax.set_title("The noise floor is measured, not chosen", fontsize=12, loc="left")
    return _save(fig, path)


def tier_triangles(designs: Sequence[dict[str, Any]], path: Path) -> Path:
    """Draw one tier triangle per design, in the task description's style.

    Each triangle has one vertex per judged property. A filled vertex marks a
    property predicted to beat trastuzumab; ties and regressions stay hollow.
    These are predictions, not Adaptyv measurements, and the figure says so.

    Args:
        designs: Dicts with ``design_id`` and boolean ``affinity``, ``epitope``
            and ``stability`` keys.
        path: Destination PNG.

    Returns:
        The path written.
    """
    n = len(designs)
    cols = min(5, max(n, 1))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.5 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    angles = np.deg2rad([90, 210, 330])
    labels = ("affinity", "epitope", "stability")

    for ax, design in zip(axes_flat, designs):
        ax.set_aspect("equal")
        ax.axis("off")
        xs, ys = np.cos(angles), np.sin(angles)
        ax.plot(
            np.append(xs, xs[0]),
            np.append(ys, ys[0]),
            color=GRID,
            linewidth=1.4,
            zorder=2,
        )
        states = [bool(design.get(k)) for k in ("affinity", "epitope", "stability")]
        tier = sum(states)
        for x, y, on, label in zip(xs, ys, states, labels):
            ax.scatter(
                [x],
                [y],
                s=110,
                zorder=3,
                facecolor=GOOD if on else SURFACE,
                edgecolor=GOOD if on else TEXT_SECONDARY,
                linewidth=1.6,
            )
            ax.text(
                x * 1.5,
                y * 1.45,
                label,
                fontsize=7,
                ha="center",
                va="center",
                color=TEXT_SECONDARY,
            )
        ax.text(
            0,
            -0.08,
            str(tier),
            fontsize=17,
            ha="center",
            va="center",
            color=TEXT_PRIMARY,
            weight="bold",
        )
        ax.set_title(
            str(design.get("design_id", ""))[:22], fontsize=7.5, color=TEXT_SECONDARY
        )
        ax.set_xlim(-1.9, 1.9)
        ax.set_ylim(-1.7, 1.9)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(
        "Predicted tier per design - filled vertex beats the baseline",
        fontsize=12,
        color=TEXT_PRIMARY,
        y=1.0,
    )
    fig.text(
        0.5,
        -0.01,
        "Predictions from this pipeline, not Adaptyv measurements.",
        ha="center",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    return _save(fig, path)


def disulfide_geometry(sites: Sequence[dict[str, Any]], path: Path) -> Path:
    """Draw candidate disulfide sites against the ideal geometry.

    Args:
        sites: Dicts with ``cb_distance``, ``conservative`` and residue labels.
        path: Destination PNG.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 3.6))
    _style(ax)
    ax.axvspan(3.5, 4.1, color=GOOD, alpha=0.09, zorder=1)
    ax.text(3.8, 0.75, "ideal window", fontsize=8.5, color=GOOD, ha="center")

    for i, site in enumerate(sites):
        conservative = bool(site.get("conservative"))
        ax.scatter(
            [site["cb_distance"]],
            [i],
            s=90,
            zorder=3,
            color=SERIES[2] if conservative else SERIES[1],
            label=(
                ("conservative substitution" if conservative else "deletes packing")
                if i == 0 or i == 1
                else None
            ),
        )
        label = (
            f"H{site['heavy_residue']}{site['heavy_seqid']}"
            f"-L{site['light_residue']}{site['light_seqid']}"
        )
        ax.text(
            site["cb_distance"] + 0.02,
            i,
            f"  {label}",
            fontsize=8.5,
            va="center",
            color=TEXT_PRIMARY,
        )

    ax.set_yticks([])
    ax.set_xlabel("C-beta to C-beta distance (angstroms)")
    ax.set_title("Disulfide sites are scanned, not assumed", fontsize=12, loc="left")
    ax.set_ylim(-0.8, max(len(sites) - 0.2, 1))
    ax.set_xlim(3.3, 4.9)
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
        for text in legend.get_texts():
            text.set_color(TEXT_SECONDARY)
    return _save(fig, path)
