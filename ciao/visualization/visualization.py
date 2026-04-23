"""Visualization functions for CIAO explanation results."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from ciao.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD


if TYPE_CHECKING:
    from ciao.explainer.ciao_explainer import ExplanationResult


_IMAGENET_MEAN = np.asarray(IMAGENET_MEAN, dtype=np.float32)
_IMAGENET_STD = np.asarray(IMAGENET_STD, dtype=np.float32)


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize an image tensor to a displayable float32 [H, W, 3] array."""
    img = tensor.squeeze(0).cpu().float().numpy().transpose(1, 2, 0)
    return np.clip(img * _IMAGENET_STD + _IMAGENET_MEAN, 0.0, 1.0)


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
    """Side-by-side: segmentation | segment-score heatmap | replacement image."""
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
