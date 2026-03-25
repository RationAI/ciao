"""Shared utilities for MCTS and MCGS search algorithms.

This module contains common functions used by both Monte Carlo Tree Search (MCTS)
and Monte Carlo Graph Search (MCGS) implementations.
"""

from collections.abc import Set

import torch

from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import calculate_hyperpixel_deltas


def is_terminal(
    current_region: Set[int],
    image_graph: ImageGraph,
    used_segments: Set[int],
    max_depth: int,
) -> bool:
    """Check if state is terminal (max depth or no frontier)."""
    return len(current_region) >= max_depth or not image_graph.get_frontier(
        current_region, used_segments
    )


def evaluate_regions(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    target_class_idx: int,
    regions: list[Set[int]],
    replacement_image: torch.Tensor,
) -> list[float]:
    """Evaluate multiple segment sets by computing class score deltas (batched)."""
    # Guard against invalid segment sets (empty)
    if any(not region for region in regions):
        raise ValueError(
            "Cannot evaluate empty segment set: A set must contain at least one segment."
        )

    all_segment_ids = [list(region) for region in regions]

    rewards = calculate_hyperpixel_deltas(
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        replacement_image=replacement_image,
        target_class_idx=target_class_idx,
        hyperpixel_segment_ids_list=all_segment_ids,
    )

    return rewards
