"""Shared utilities for MCTS and MCGS search algorithms.

This module contains common functions used by both Monte Carlo Tree Search (MCTS)
and Monte Carlo Graph Search (MCGS) implementations.
"""

import numpy as np
import torch

from ciao.algorithm.bitmask_graph import get_frontier, iter_bits
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import calculate_hyperpixel_deltas


def is_terminal(
    mask: int, adj_masks: tuple[int, ...], used_mask: int, max_depth: int
) -> bool:
    """Check if state is terminal (max depth or no frontier)."""
    return (
        mask.bit_count() >= max_depth or get_frontier(mask, adj_masks, used_mask) == 0
    )


def evaluate_masks(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    target_class_idx: int,
    masks: list[int],
    replacement_image: torch.Tensor,
) -> list[float]:
    """Evaluate multiple segment masks by computing class score deltas (batched)."""
    # Guard against invalid masks (zero or negative)
    if any(mask <= 0 for mask in masks):
        raise ValueError(
            "Cannot evaluate invalid mask: A mask must be a positive integer. "
            "Zero masks contain no segments, and negative masks cause "
            "incorrect bit iteration due to two's complement representation."
        )

    all_segment_ids = [list(iter_bits(mask)) for mask in masks]

    rewards = calculate_hyperpixel_deltas(
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        replacement_image=replacement_image,
        target_class_idx=target_class_idx,
        hyperpixel_segment_ids_list=all_segment_ids,
    )

    return rewards
