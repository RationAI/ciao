"""Evaluation metrics for CIAO explanations."""

from ciao.metrics.saliency import (
    build_saliency_map,
    compute_deletion_auc,
    compute_deletion_curve,
    compute_insertion_auc,
    compute_insertion_curve,
    compute_iou_top_fraction,
    compute_pointing_game,
)
from ciao.metrics.segmentation import compute_iou


__all__ = [
    "build_saliency_map",
    "compute_deletion_auc",
    "compute_deletion_curve",
    "compute_insertion_auc",
    "compute_insertion_curve",
    "compute_iou",
    "compute_iou_top_fraction",
    "compute_pointing_game",
]
