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
    # TODO: change so that we sample while num of unique regions < num_simulations
    # maybe rename num_simulations to sth like target_evals
    # add a parameter for patience - if we already found `patience` duplicates, stop with simulations. Probably hyperparam

    unique_regions = list(dict.fromkeys(sampled_regions))
    if not unique_regions:
        unique_regions = [seed_region]

    # TODO: evaluate in batches - I want the trajectory to be realistic, not just one number
    # so maybe even add it to the sampling while cycle and every batch_size times call this function
    scores = calculate_region_deltas(
        predictor=ctx.predictor,
        input_batch=ctx.input_batch,
        segments=ctx.image_graph.segments,
        replacement_image=ctx.replacement_image,
        segment_sets=unique_regions,
        target_class_idx=ctx.target_class_idx,
        original_log_odds=ctx.original_log_odds,
        batch_size=ctx.batch_size,
    )

    signed_scores = [s * ctx.optimization_sign for s in scores]

    best_signed = -float("inf")
    trajectory: list[dict[str, float]] = []
    for idx, signed in enumerate(signed_scores, start=1):
        if signed > best_signed:
            best_signed = signed
        trajectory.append({"evals": idx, "best_score": best_signed})

    best_idx = max(range(len(signed_scores)), key=lambda i: signed_scores[i])

    return RegionResult(
        region=unique_regions[best_idx],
        score=scores[best_idx],
        evaluations_count=len(unique_regions),
        trajectory=trajectory,
    )
