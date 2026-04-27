"""Potential algorithm — Sequential Monte Carlo with Potential-based Selection.

Uses set for regions.
"""

from collections.abc import Set

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_potential(
    ctx: SearchContext,
    num_simulations: int,
) -> RegionResult:
    """Build a region using Sequential Monte Carlo with Potential-based Selection.

    Args:
        ctx: Search context
        num_simulations: Monte Carlo samples per frontier node per iteration

    Returns:
        RegionResult with region, score, and statistics
    """
    used_segments = set(ctx.used_segments)

    # Initialize current region structure
    curr_region = frozenset({ctx.seed_idx})

    # Cache actual scores across all steps to avoid duplicate evaluations
    evaluated_scores: dict[frozenset[int], float] = {}

    eval_count = 0
    trajectory: list[dict[str, float]] = []
    best_signed = -float("inf")

    # Main loop: Grow region until target length
    while len(curr_region) < ctx.desired_length:
        # Compute expansion frontier
        current_frontier = ctx.image_graph.get_frontier(curr_region, used_segments)

        if not current_frontier:
            break

        # Phase 1: Sampling - Monte Carlo exploration from each frontier node
        potentials, new_evals = sampling_phase(
            curr_region=curr_region,
            current_frontier=current_frontier,
            num_simulations=num_simulations,
            ctx=ctx,
            used_segments=used_segments,
            evaluated_scores=evaluated_scores,
        )
        eval_count += new_evals

        if evaluated_scores:
            step_best = max(
                s * ctx.optimization_sign for s in evaluated_scores.values()
            )
            if step_best > best_signed:
                best_signed = step_best
            trajectory.append({"evals": eval_count, "best_score": best_signed})

        # Phase 2: Selection - Choose best frontier node by potential
        valid_nodes = [n for n in potentials if potentials[n]]
        if not valid_nodes:
            break

        winner = max(valid_nodes, key=lambda n: sorted(potentials[n], reverse=True))

        # Commit: Add winner to region structure
        curr_region = curr_region | {winner}

    # Fetch final score from cache, or evaluate if loop never ran
    final_score = evaluated_scores.get(frozenset(curr_region))
    if final_score is None:
        final_score = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=[curr_region],
            target_class_idx=ctx.target_class_idx,
            batch_size=ctx.batch_size,
        )[0]
        eval_count += 1
        signed_final = final_score * ctx.optimization_sign
        if signed_final > best_signed:
            best_signed = signed_final
        trajectory.append({"evals": eval_count, "best_score": best_signed})

    return RegionResult(
        region=curr_region,
        score=final_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )


def sampling_phase(
    curr_region: Set[int],
    current_frontier: Set[int],
    num_simulations: int,
    ctx: SearchContext,
    used_segments: Set[int],
    evaluated_scores: dict[frozenset[int], float],
) -> tuple[dict[int, list[float]], int]:
    """Monte Carlo Sampling Phase: Explore expansions and populate potential cache.

    For each frontier node n:
        1. Create extended structure S U {n}
        2. Run num_simulations random walk expansions from this extended structure
        3. Evaluate each unique expansion with the model
        4. Distribute results to cache: For each frontier node that appears in an
           expansion, record (expansion_region, score) in that node's history

    Args:
        curr_region: Current region structure (set-like)
        current_frontier: Frontier nodes (set for hit detection and iteration)
        num_simulations: Number of random walk simulations per frontier node
        ctx: Search context containing target length, model state, and evaluation parameters
        used_segments: Set-like wrapper of already-used nodes
        evaluated_scores: Global cache of previously evaluated regions to their scores

    Returns:
        Tuple of (mapping from frontier node ID to a list of evaluated scores
        for expansions containing that node, number of new NN evaluations performed).
    """
    regions_to_evaluate: list[frozenset[int]] = []
    # Maps expansion_region -> which frontier nodes it contains
    region_to_frontier_hits: dict[frozenset[int], frozenset[int]] = {}

    # --- Sampling Loop: Generate candidate expansions ---
    for n in current_frontier:
        # Start with S U {n}
        extended_region = curr_region | {n}

        for _ in range(num_simulations):
            sampled_region = ctx.image_graph.sample_connected_superset(
                base_region=extended_region,
                target_length=ctx.desired_length,
                used_segments=used_segments,
            )

            if sampled_region in region_to_frontier_hits:
                continue

            # Bucketization: Which frontier nodes appear in this expansion?
            hits = sampled_region & current_frontier
            region_to_frontier_hits[sampled_region] = hits

            if sampled_region not in evaluated_scores:
                regions_to_evaluate.append(sampled_region)

    if not region_to_frontier_hits:
        return {}, 0

    # --- Batch Evaluation: Score all unique expansions ---
    if regions_to_evaluate:
        scores = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=regions_to_evaluate,
            target_class_idx=ctx.target_class_idx,
            batch_size=ctx.batch_size,
        )

        for region, score in zip(regions_to_evaluate, scores, strict=True):
            evaluated_scores[region] = score

    # --- Distribution to Potentials ---
    potentials: dict[int, list[float]] = {}
    for frozen_region, hits in region_to_frontier_hits.items():
        score = evaluated_scores[frozen_region]
        signed_score = score * ctx.optimization_sign

        # Distribute to all neighbors in the hit set
        for neighbor_id in hits:
            potentials.setdefault(neighbor_id, []).append(signed_score)

    return potentials, len(regions_to_evaluate)


# TODO: maybe add fixed num_simulations for the whole iterations? So that I can control it?

# TODO: maybe add an option to do UCB instead of the potentials

# TODO: shouldn't we track the trajectory more precisely? Inside the sampling phase?

# TODO: keep track of the best score, maybe some cache? use the evaluated_scores and do sth like redistribute_history to the best neighbor?
