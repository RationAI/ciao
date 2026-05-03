"""Visualization functions for CIAO explanation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from ciao.data.constants import IMAGENET_MEAN, IMAGENET_STD


if TYPE_CHECKING:
    from ciao.explainer.ciao_explainer import ExplanationResult


_IMAGENET_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize an image tensor to a displayable float32 [H, W, 3] array."""
    img = tensor.detach().squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)


def _to_hw(mask: torch.Tensor) -> np.ndarray:
    """Convert a soft mask tensor to a displayable float32 [H, W] array."""
    return mask.detach().squeeze().cpu().float().numpy()


def _segment_boundaries(segments: np.ndarray) -> np.ndarray:
    """Return a boolean [H, W] mask that is True on segment edges."""
    h_edge = np.pad(segments[:-1] != segments[1:], ((0, 1), (0, 0)))
    v_edge = np.pad(segments[:, :-1] != segments[:, 1:], ((0, 0), (0, 1)))
    return h_edge | v_edge


def _region_mask(segments: np.ndarray, region: frozenset[int]) -> np.ndarray:
    """Return a boolean [H, W] mask covering all pixels in *region*."""
    mask = np.zeros(segments.shape, dtype=bool)
    for seg_id in region:
        mask |= segments == seg_id
    return mask


def plot_overview(result: ExplanationResult) -> Figure:
    """Side-by-side: original | segmentation | segment-score heatmap | replacement image."""
    img = _to_hwc(result.input_batch)
    segs = result.segments.cpu().numpy()
    repl = _to_hwc(result.replacement_image.unsqueeze(0))

    boundaries = _segment_boundaries(segs)
    seg_overlay = img.copy()
    seg_overlay[boundaries] = 1.0

    score_map = np.zeros(segs.shape, dtype=np.float32)
    for seg_id, score in result.segment_scores.items():
        score_map[segs == seg_id] = score
    abs_max = float(np.abs(score_map).max()) or 1.0

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(img)
    axes[0].set_title("original")
    axes[0].axis("off")

    axes[1].imshow(seg_overlay)
    axes[1].set_title("segmentation")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(score_map, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, alpha=0.55)
    axes[2].set_title("segment scores")
    axes[2].axis("off")

    axes[3].imshow(repl)
    axes[3].set_title("replacement")
    axes[3].axis("off")

    fig.tight_layout(pad=0.5)
    return fig


def plot_regions(result: ExplanationResult) -> Figure:
    """One subplot per region: region pixels replaced, rest is original."""
    img = _to_hwc(result.input_batch)
    repl = _to_hwc(result.replacement_image.unsqueeze(0))
    segs = result.segments.cpu().numpy()
    n = len(result.regions)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, region_result in zip(axes[0], result.regions, strict=True):
        mask = _region_mask(segs, region_result.region)
        blended = img.copy()
        blended[mask] = repl[mask]
        ax.imshow(blended)
        ax.axis("off")

    fig.tight_layout(pad=0)
    return fig


def plot_region_scores(result: ExplanationResult) -> Figure:
    """One subplot per region: region pixels tinted by score, rest is original.

    Positive score → red tint, negative → blue tint (diverging, symmetric).
    """
    img = _to_hwc(result.input_batch)
    segs = result.segments.cpu().numpy()
    n = len(result.regions)

    all_scores = [r.score for r in result.regions]
    abs_max = max(abs(s) for s in all_scores) or 1.0
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, region_result in zip(axes[0], result.regions, strict=True):
        mask = _region_mask(segs, region_result.region)
        tint = np.array(cmap(norm(region_result.score))[:3], dtype=np.float32)

        colored = img.copy()
        colored[mask] = colored[mask] * 0.35 + tint * 0.65

        ax.imshow(colored)
        ax.axis("off")

    fig.tight_layout(pad=0)
    return fig


def plot_soft_mask(result: ExplanationResult) -> Figure:
    """Show the EP soft mask as a standalone heatmap and an overlay."""
    if result.soft_mask is None:
        raise ValueError("soft_mask is not available for this explanation")

    img = _to_hwc(result.input_batch)
    mask = _to_hw(result.soft_mask)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(mask, cmap="magma", vmin=0.0, vmax=1.0)
    axes[0].set_title("soft mask")
    axes[0].axis("off")

    axes[1].imshow(img)
    axes[1].imshow(mask, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.6)
    axes[1].set_title("soft mask overlay")
    axes[1].axis("off")

    fig.tight_layout(pad=0.5)
    return fig


def plot_heatmap_overlay(result: ExplanationResult, alpha: float = 0.5) -> Figure:
    """Single image: heatmap overlaid on the original. No title, no axes, no decorations.

    Works for any baseline that sets ``soft_mask`` (GradCAM, Occlusion, MP, LIME).
    The heatmap is min-max normalised to [0, 1] before display.

    Args:
        result: ExplanationResult with ``soft_mask`` populated.
        alpha: Opacity of the heatmap overlay.
    """
    if result.soft_mask is None:
        raise ValueError("soft_mask is not available for this explanation")

    img = _to_hwc(result.input_batch)
    mask = _to_hw(result.soft_mask)

    mn, mx = float(mask.min()), float(mask.max())
    mask_norm = (mask - mn) / (mx - mn) if mx > mn else np.zeros_like(mask)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(img)
    ax.imshow(mask_norm, cmap="jet", alpha=alpha, vmin=0.0, vmax=1.0)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def plot_deletion_curve(
    fractions: np.ndarray,
    probs: np.ndarray,
    auc: float | None = None,
    class_name: str | None = None,
) -> Figure:
    """Plot the deletion curve: target class probability vs fraction of pixels deleted."""
    if auc is None:
        auc = float(np.trapezoid(probs, fractions))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fractions, probs, color="steelblue", linewidth=2)
    ax.fill_between(
        fractions, probs, alpha=0.2, color="steelblue", label=f"AUC = {auc:.4f}"
    )
    ax.set_xlabel("fraction of pixels deleted")
    ax.set_ylabel("target class probability")
    title = "deletion curve" if class_name is None else f"deletion curve — {class_name}"
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_insertion_curve(
    fractions: np.ndarray,
    probs: np.ndarray,
    auc: float | None = None,
    class_name: str | None = None,
) -> Figure:
    """Plot the insertion curve: target class probability vs fraction of pixels revealed.

    Args:
        fractions: 1-D array of revealed pixel fractions (x-axis, 0 to 1).
        probs: 1-D array of target class probabilities at each fraction (y-axis).
        auc: Pre-computed AUC to annotate on the plot (computed if not provided).
        class_name: Optional label shown in the figure title.
    """
    if auc is None:
        auc = float(np.trapezoid(probs, fractions))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fractions, probs, color="darkorange", linewidth=2)
    ax.fill_between(
        fractions, probs, alpha=0.2, color="darkorange", label=f"AUC = {auc:.4f}"
    )
    ax.set_xlabel("fraction of pixels revealed")
    ax.set_ylabel("target class probability")
    title = (
        "insertion curve" if class_name is None else f"insertion curve — {class_name}"
    )
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    return fig
