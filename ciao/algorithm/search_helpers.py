"""Shared utilities for region-search algorithms."""

from collections.abc import Set as AbstractSet

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.scoring.region import calculate_region_deltas


def is_terminal(
    current_region: AbstractSet[int],
    image_graph: ImageGraph,
    used_segments: AbstractSet[int],
    max_depth: int,
) -> bool:
    """Check if state is terminal (max depth or no frontier)."""
    return len(current_region) >= max_depth or not image_graph.get_frontier(
        current_region, used_segments
    )


def evaluate_and_cache(
    regions_to_evaluate: list[frozenset[int]],
    evaluated_scores: dict[frozenset[int], float],
    ctx: SearchContext,
) -> int:
    """Score uncached regions in one batched pass and write into ``evaluated_scores``.

    Args:
        regions_to_evaluate: Unique regions not yet present in ``evaluated_scores``.
        evaluated_scores: Global cache mutated with the new ``region -> score`` entries.
        ctx: Search context providing predictor, segments, replacement, etc.

    Returns:
        Number of new NN evaluations performed (i.e. ``len(regions_to_evaluate)``).
    """
    if not regions_to_evaluate:
        return 0

    scores = calculate_region_deltas(
        predictor=ctx.predictor,
        input_batch=ctx.input_batch,
        segments=ctx.image_graph.segments,
        replacement_image=ctx.replacement_image,
        segment_sets=regions_to_evaluate,
        target_class_idx=ctx.target_class_idx,
        original_log_odds=ctx.original_log_odds,
        batch_size=ctx.batch_size,
    )

    for region, score in zip(regions_to_evaluate, scores, strict=True):
        evaluated_scores[region] = score

    return len(regions_to_evaluate)
