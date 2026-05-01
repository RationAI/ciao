"""Potential algorithm — Sequential Monte Carlo with Potential-based Selection.

Uses set for regions.
"""

import time
from collections.abc import Set

from ciao.algorithm.context import SearchContext
from ciao.algorithm.search_helpers import evaluate_and_cache
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_potential(
    ctx: SearchContext,
    step_budget: int,
) -> RegionResult:
    """Build a region using Sequential Monte Carlo with Potential-based Selection.

    Args:
        ctx: Search context
        step_budget: Maximum number of rollouts per commit step (same semantics as UCB)

    Returns:
        RegionResult with region, score, and statistics. The returned region is
        the best-scoring region encountered during the search (across all rollouts
        and committed states), not necessarily the final committed region.
    """
    used_segments = set(ctx.used_segments)

    # Initialize current region structure
    curr_region = frozenset({ctx.seed_idx})

    # Cache actual scores across all steps to avoid duplicate evaluations
    evaluated_scores: dict[frozenset[int], float] = {}

    eval_count = 0
    trajectory: list[dict[str, float]] = []
    best_signed = -float("inf")
    best_region: frozenset[int] = curr_region
    best_raw_score: float = 0.0
    t0 = time.monotonic()

    # Main loop: Grow region until target length
    while len(curr_region) < ctx.desired_length:
        # Compute expansion frontier
        current_frontier = ctx.image_graph.get_frontier(curr_region, used_segments)

        if not current_frontier:
            break

        # Phase 1: Sampling - Monte Carlo exploration from each frontier node
        potentials, new_evals, step_best = sampling_phase(
            curr_region=curr_region,
            current_frontier=current_frontier,
            step_budget=step_budget,
            ctx=ctx,
            used_segments=used_segments,
            evaluated_scores=evaluated_scores,
        )
        eval_count += new_evals

        if step_best is not None:
            signed, region, raw = step_best
            if signed > best_signed:
                best_signed = signed
                best_region = region
                best_raw_score = raw
            trajectory.append(
                {
                    "evals": eval_count,
                    "best_score": best_signed,
                    "time": time.monotonic() - t0,
                }
            )

        # Phase 2: Selection - Choose best frontier node by potential
        valid_nodes = [n for n in potentials if potentials[n]]
        if not valid_nodes:
            break

        winner = max(valid_nodes, key=lambda n: sorted(potentials[n], reverse=True))

        # Commit: Add winner to region structure
        curr_region = curr_region | {winner}

    # If the loop never produced any evaluation (e.g. immediate dead end),
    # score the seed-only region so we have something to return.
    if best_signed == -float("inf"):
        score = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=[curr_region],
            target_class_idx=ctx.target_class_idx,
            original_log_odds=ctx.original_log_odds,
            batch_size=ctx.batch_size,
        )[0]
        eval_count += 1
        best_signed = score * ctx.optimization_sign
        best_region = curr_region
        best_raw_score = score
        trajectory.append(
            {
                "evals": eval_count,
                "best_score": best_signed,
                "time": time.monotonic() - t0,
            }
        )

    return RegionResult(
        region=best_region,
        score=best_raw_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )


def sampling_phase(
    curr_region: Set[int],
    current_frontier: Set[int],
    step_budget: int,
    ctx: SearchContext,
    used_segments: Set[int],
    evaluated_scores: dict[frozenset[int], float],
) -> tuple[
    dict[int, list[float]],
    int,
    tuple[float, frozenset[int], float] | None,
]:
    """Monte Carlo Sampling Phase: Explore expansions and populate potential cache.

    Distributes step_budget rollouts across frontier nodes (round-robin), then
    evaluates unique expansions and scores each frontier node by the potentials
    of expansions that contain it.

    Args:
        curr_region: Current region structure (set-like)
        current_frontier: Frontier nodes (set for hit detection and iteration)
        step_budget: Total number of rollouts for this commit step (same as UCB)
        ctx: Search context containing target length, model state, and evaluation parameters
        used_segments: Set-like wrapper of already-used nodes
        evaluated_scores: Global cache of previously evaluated regions to their scores

    Returns:
        Tuple of (mapping from frontier node ID to a list of evaluated scores
        for expansions containing that node, number of new NN evaluations
        performed, optional (signed_score, region, raw_score) for the best
        region observed across the cache after this step).
    """
    regions_to_evaluate: list[frozenset[int]] = []
    # Maps expansion_region -> which frontier nodes it contains
    region_to_frontier_hits: dict[frozenset[int], frozenset[int]] = {}

    frontier_list = list(current_frontier)
    extended_regions = {n: frozenset(curr_region | {n}) for n in frontier_list}

    # --- Sampling Loop: Distribute step_budget rollouts round-robin ---
    for i in range(step_budget):
        n = frontier_list[i % len(frontier_list)]
        sampled_region = ctx.image_graph.sample_connected_superset(
            base_region=extended_regions[n],
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
        return {}, 0, None

    # --- Batch Evaluation: Score all unique expansions ---
    new_evals = evaluate_and_cache(regions_to_evaluate, evaluated_scores, ctx)

    # --- Distribution to Potentials ---
    potentials: dict[int, list[float]] = {}
    best_signed = -float("inf")
    best_region: frozenset[int] | None = None
    best_raw: float = 0.0
    for frozen_region, hits in region_to_frontier_hits.items():
        score = evaluated_scores[frozen_region]
        signed_score = score * ctx.optimization_sign

        if signed_score > best_signed:
            best_signed = signed_score
            best_region = frozen_region
            best_raw = score

        # Distribute to all neighbors in the hit set
        for neighbor_id in hits:
            potentials.setdefault(neighbor_id, []).append(signed_score)

    step_best = (
        (best_signed, best_region, best_raw) if best_region is not None else None
    )
    return potentials, new_evals, step_best
