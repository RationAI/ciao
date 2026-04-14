"""Unified region builder orchestrating different search algorithms."""

from typing import Literal

import torch

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor
from ciao.scoring.region import RegionResult, calculate_region_probability_drops
from ciao.typing import ExplanationMethodFn


SeedSelectionMode = Literal[-1, 1] | None


def _select_seed_and_sign(
    scores: dict[int, float], available_segments: list[int], sigma: SeedSelectionMode
) -> tuple[int, int]:
    if not available_segments:
        raise ValueError(
            "Cannot select a seed from an empty set of available segments."
        )

    if sigma is None:
        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        optimization_sign = 1 if scores[seed_idx] >= 0 else -1
    elif sigma == 1:
        seed_idx = max(available_segments, key=lambda x: scores[x])
        optimization_sign = 1
    elif sigma == -1:
        seed_idx = max(available_segments, key=lambda x: -scores[x])
        optimization_sign = -1
    else:
        raise ValueError(f"sigma must be -1, 1, or 'auto', got {sigma!r}")

    return seed_idx, optimization_sign


def build_all_regions(
    method: ExplanationMethodFn,
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    replacement_image: torch.Tensor,
    image_graph: ImageGraph,
    target_class_idx: int,
    original_log_odds: torch.Tensor,
    scores: dict[int, float],
    max_regions: int,
    original_prob: float,
    sigma: SeedSelectionMode = None,
    desired_length: int = 30,
    batch_size: int = 64,
) -> list[RegionResult]:
    """Build multiple regions using the provided single-region algorithm.

    This function handles the outer loop for all search algorithms:
    finding seeds, calling the specific algorithm, updating used segments,
    and sorting the final results.

    Args:
        method: A callable strategy for constructing a single region.
        predictor: Model predictor
        input_batch: Preprocessed image batch
        replacement_image: Replacement tensor
        image_graph: Graph representation of image segments
        target_class_idx: Target class index
        original_log_odds: Pre-computed unmasked target-class log-odds (scalar tensor)
        scores: Base segment scores
        max_regions: Maximum number of regions to construct
        original_prob: Pre-computed unmasked probability for the target class
        sigma: Seed selection mode. ``"None"`` picks max abs score and then inherits its sign. ``1`` chooses the
            highest raw score (positive evidence), ``-1`` chooses the lowest
            raw score (negative evidence).
        desired_length: Target number of segments per region
        batch_size: Batch size for model evaluation

    Returns:
        List[RegionResult]: list of RegionResult objects sorted by absolute score
    """
    regions: list[RegionResult] = []
    used_segments: set[int] = set()

    for _ in range(max_regions):
        # Find best unprocessed seed
        available_segments = [
            seg_id for seg_id in scores if seg_id not in used_segments
        ]

        if not available_segments:
            break

        seed_idx, optimization_sign = _select_seed_and_sign(
            scores=scores, available_segments=available_segments, sigma=sigma
        )

        # Construct a SearchContext for the current step
        ctx = SearchContext(
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            original_log_odds=original_log_odds,
            segment_scores=scores,
            seed_idx=seed_idx,
            optimization_sign=optimization_sign,
            used_segments=frozenset(used_segments),
            desired_length=desired_length,
            batch_size=batch_size,
        )

        # Call the dynamically provided algorithm for a single region
        result = method(ctx)

        current_region = result.region

        if not current_region:
            raise RuntimeError(
                f"Builder failed to generate any segments for seed {seed_idx}."
            )

        # Extract and update state
        used_segments = used_segments | current_region
        regions.append(result)

    # Compute probability drops for all finished regions in batched forward passes
    calculate_region_probability_drops(
        predictor=predictor,
        input_batch=input_batch,
        segments=image_graph.segments,
        replacement_image=replacement_image,
        target_class_idx=target_class_idx,
        original_prob=original_prob,
        results=regions,
        batch_size=batch_size,
    )

    # Sort by absolute score
    regions.sort(key=lambda x: abs(x.score), reverse=True)

    return regions
