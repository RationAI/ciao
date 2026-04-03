"""Unified hyperpixel builder orchestrating different search algorithms."""

import torch

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.explainer.methods import ExplanationMethod, LookaheadMethod
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import HyperpixelResult


def build_all_hyperpixels(
    method: ExplanationMethod | None,
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    replacement_image: torch.Tensor,
    image_graph: ImageGraph,
    target_class_idx: int,
    scores: dict[int, float],
    max_hyperpixels: int,
    desired_length: int = 30,
    batch_size: int = 64,
) -> list[HyperpixelResult]:
    """Build multiple hyperpixels using the provided single-hyperpixel algorithm.

    This function handles the outer loop for all search algorithms:
    finding seeds, calling the specific algorithm, updating used segments,
    and sorting the final results.

    Args:
        method: A callable strategy for constructing a single hyperpixel. Default is LookaheadMethod.
        predictor: Model predictor
        input_batch: Preprocessed image batch
        replacement_image: Replacement tensor
        image_graph: Graph representation of image segments
        target_class_idx: Target class index
        scores: Base segment scores
        max_hyperpixels: Maximum number of hyperpixels to construct
        desired_length: Target number of segments per hyperpixel
        batch_size: Batch size for model evaluation

    Returns:
        List[HyperpixelResult]: list of HyperpixelResult objects sorted by absolute score
    """
    if method is None:
        method = LookaheadMethod()

    hyperpixels: list[HyperpixelResult] = []
    used_segments: set[int] = set()

    for _ in range(max_hyperpixels):
        # Find best unprocessed seed
        available_segments = [
            seg_id for seg_id in scores if seg_id not in used_segments
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        # Construct a SearchContext for the current step
        ctx = SearchContext(
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            seed_idx=seed_idx,
            optimization_sign=optimization_sign,
            used_segments=frozenset(used_segments),
            desired_length=desired_length,
            batch_size=batch_size,
        )

        # Call the dynamically provided algorithm for a single hyperpixel
        result = method(ctx)

        hyperpixel_region = result.region

        if not hyperpixel_region:
            raise RuntimeError(
                f"Builder failed to generate any segments for seed {seed_idx}."
            )

        # Extract and update state
        used_segments = used_segments | hyperpixel_region
        hyperpixels.append(result)

    # Sort by absolute score
    hyperpixels.sort(key=lambda x: abs(x.score), reverse=True)

    return hyperpixels
