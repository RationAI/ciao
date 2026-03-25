"""Unified hyperpixel builder orchestrating different search algorithms."""

import logging
from collections.abc import Callable
from typing import Any

import torch

from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import HyperpixelResult


logger = logging.getLogger(__name__)


def build_all_hyperpixels(
    builder_func: Callable[..., HyperpixelResult],
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    replacement_image: torch.Tensor,
    image_graph: ImageGraph,
    target_class_idx: int,
    scores: dict[int, float],
    max_hyperpixels: int,
    **algo_kwargs: Any,
) -> list[HyperpixelResult]:
    """Build multiple hyperpixels using the provided single-hyperpixel algorithm.

    This function handles the outer loop for all search algorithms:
    finding seeds, calling the specific algorithm, updating used segments,
    and sorting the final results.

    Args:
        builder_func: Callable that builds a single hyperpixel (e.g., MCTS, Greedy).
        predictor: Model predictor
        input_batch: Preprocessed image batch
        segments: Segmentation map
        replacement_image: Replacement tensor
        image_graph: Graph representation of image segments
        target_class_idx: Target class index
        scores: Base segment scores
        max_hyperpixels: Maximum number of hyperpixels to construct
        **algo_kwargs: Algorithm-specific parameters passed to builder_func

    Returns:
        List of hyperpixel dicts sorted by absolute score
    """
    hyperpixels = []
    processed_segments: set[int] = set()
    used_segments: frozenset[int] = frozenset()

    for i in range(max_hyperpixels):
        # Find best unprocessed seed
        available_segments = [
            seg_id for seg_id in scores if seg_id not in processed_segments
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        logger.info(f"\n--- Hyperpixel {i + 1}/{max_hyperpixels} ---")
        logger.info(
            f"Seed: {seed_idx}, score: {seed_score:.4f}, sign: {optimization_sign}"
        )

        # Call the dynamically provided algorithm for a single hyperpixel
        result = builder_func(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            seed_idx=seed_idx,
            optimization_sign=optimization_sign,
            used_segments=used_segments,
            **algo_kwargs,
        )

        # Extract and update state
        hyperpixel_region = result["region"]
        used_segments = frozenset(used_segments | hyperpixel_region)

        if not hyperpixel_region:
            raise RuntimeError(
                f"Builder failed to generate any segments for seed {seed_idx}."
            )

        hyperpixels.append(result)
        processed_segments.update(hyperpixel_region)

    # Sort by absolute score
    hyperpixels.sort(key=lambda x: abs(float(x["score"])), reverse=True)

    return hyperpixels
