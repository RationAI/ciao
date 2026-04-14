"""Pure Monte Carlo search for connected image regions."""

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_pure_monte_carlo(
    ctx: SearchContext,
    num_simulations: int,
) -> RegionResult:
    """Build a region via random sampling from the seed and pick the best sample.

    Args:
        ctx: Search context with model state and search parameters.
        num_simulations: Number of random connected supersets to sample.

    Returns:
        RegionResult containing the best sampled region and its score.
    """
    if num_simulations < 1:
        raise ValueError(f"num_simulations must be >= 1, got {num_simulations}")

    seed_region = frozenset({ctx.seed_idx})
    sampled_regions = [
        ctx.image_graph.sample_connected_superset(
            base_region=seed_region,
            target_length=ctx.desired_length,
            used_segments=ctx.used_segments,
        )
        for _ in range(num_simulations)
    ]

    unique_regions = list(dict.fromkeys(sampled_regions))
    if not unique_regions:
        unique_regions = [seed_region]

    scores = calculate_region_deltas(
        predictor=ctx.predictor,
        input_batch=ctx.input_batch,
        segments=ctx.image_graph.segments,
        replacement_image=ctx.replacement_image,
        segment_sets=unique_regions,
        target_class_idx=ctx.target_class_idx,
        batch_size=ctx.batch_size,
    )

    best_idx = max(
        range(len(scores)),
        key=lambda idx: scores[idx] * ctx.optimization_sign,
    )

    return RegionResult(region=unique_regions[best_idx], score=scores[best_idx])
