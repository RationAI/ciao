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
    """Single image: all regions replaced at once."""
    img = _to_hwc(result.input_batch)
    repl = _to_hwc(result.replacement_image.unsqueeze(0))
    segs = result.segments.cpu().numpy()

    composite = img.copy()
    for region_result in result.regions:
        mask = _region_mask(segs, region_result.region)
        composite[mask] = repl[mask]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(composite)
    ax.axis("off")
    fig.tight_layout(pad=0)
    return fig


def plot_saliency_map(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    class_name: str | None = None,
) -> Figure:
    """Side-by-side: original image | saliency map overlaid on the image.

    Args:
        image: [1, C, H, W] or [C, H, W] preprocessed image tensor.
        saliency_map: [H, W] float array in [0, 1], higher = more salient.
        class_name: Optional label shown in the figure title.
    """
    img = _to_hwc(image if image.dim() == 4 else image.unsqueeze(0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].set_title("original" if class_name is None else class_name)
    axes[0].axis("off")

    axes[1].imshow(img)
    sal = axes[1].imshow(saliency_map, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    axes[1].set_title("saliency map")
    axes[1].axis("off")
    fig.colorbar(sal, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout(pad=0.5)
    return fig


def plot_deletion_curve(
    fractions: np.ndarray,
    probs: np.ndarray,
    auc: float | None = None,
    class_name: str | None = None,
) -> Figure:
    """Plot the deletion curve: target class probability vs fraction of pixels deleted.

    Args:
        fractions: 1-D array of masked pixel fractions (x-axis, 0 to 1).
        probs: 1-D array of target class probabilities at each fraction (y-axis).
        auc: Pre-computed AUC to annotate on the plot (computed if not provided).
        class_name: Optional label shown in the figure title.
    """
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


def plot_region_scores(result: ExplanationResult) -> Figure:
    """Single image: all regions tinted by score with cycling opacity.

    Positive score → red tint, negative → blue tint (diverging, symmetric).
    Opacity cycles across regions so adjacent regions remain visually distinct.
    """
    img = _to_hwc(result.input_batch)
    segs = result.segments.cpu().numpy()

    all_scores = [r.score for r in result.regions]
    abs_max = max(abs(s) for s in all_scores) or 1.0
    cmap = plt.get_cmap("RdBu_r")
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    alphas = [0.45, 0.65, 0.85]

    composite = img.copy()
    for i, region_result in enumerate(result.regions):
        mask = _region_mask(segs, region_result.region)
        tint = np.array(cmap(norm(region_result.score))[:3], dtype=np.float32)
        alpha = alphas[i % len(alphas)]
        composite[mask] = composite[mask] * (1 - alpha) + tint * alpha

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(composite)
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
