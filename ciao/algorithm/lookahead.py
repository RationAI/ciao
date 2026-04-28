"""Greedy lookahead region building with frozensets.

Rolling horizon strategy: Look ahead multiple steps but only commit one step at a time.
"""

from collections import deque
from collections.abc import Set

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_greedy_lookahead(
    ctx: SearchContext,
    lookahead_distance: int,
) -> RegionResult:
    """Build a single region using greedy lookahead with rolling horizon.

    Strategy: Look ahead up to lookahead_distance steps, evaluate all candidates,
    but only commit the first step of the best path found.

    Args:
        ctx: Search context
        lookahead_distance: How many steps to look ahead (1=greedy, 2+=lookahead)

    Raises:
        ValueError: If lookahead_distance is less than 1.

    Returns:
        RegionResult containing region and score
    """
    if lookahead_distance < 1:
        raise ValueError(f"lookahead_distance must be >= 1, got {lookahead_distance}")

    desired_length = ctx.desired_length
    optimization_sign = ctx.optimization_sign
    seed_idx = ctx.seed_idx
    image_graph = ctx.image_graph
    used_segments = ctx.used_segments

    current_region = frozenset([seed_idx])
    known_final_score: float | None = None

    eval_count = 0
    trajectory: list[dict[str, float]] = []
    current_best_score = -float("inf")

    # Grow region one step at a time
    while len(current_region) < desired_length:
        # Generate all candidate regions via BFS up to lookahead_distance
        candidates = _generate_lookahead_candidates(
            current_region=current_region,
            image_graph=image_graph,
            used_segments=used_segments,
            lookahead_distance=lookahead_distance,
            desired_length=desired_length,
        )

        if not candidates:
            break

        # Batch evaluate all candidates
        candidate_regions = list(candidates.keys())

        scores_list = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=image_graph.segments,
            segment_sets=candidate_regions,
            replacement_image=ctx.replacement_image,
            target_class_idx=ctx.target_class_idx,
            batch_size=ctx.batch_size,
            original_log_odds=ctx.original_log_odds,
        )
        eval_count += len(candidate_regions)

        # Find best candidate (maximize optimization_sign * score)
        best_idx = max(
            range(len(scores_list)), key=lambda i: scores_list[i] * optimization_sign
        )
        best_region = candidate_regions[best_idx]
        best_score = scores_list[best_idx]
        first_step = candidates[best_region]

        signed_best = best_score * optimization_sign
        if signed_best > current_best_score:
            current_best_score = signed_best
        trajectory.append({"evals": eval_count, "best_score": current_best_score})

        # Optimization - commit entire path
        if len(best_region) == desired_length:
            current_region = best_region
            known_final_score = best_score
            break

        # Commit only the first step
        current_region = current_region | {first_step}

    # Re-evaluate the final built region to get its exact score.
    if known_final_score is not None:
        final_score = known_final_score
    # This could happen if we exhausted all candidates before reaching desired_length
    else:
        final_score = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=image_graph.segments,
            segment_sets=[current_region],
            replacement_image=ctx.replacement_image,
            target_class_idx=ctx.target_class_idx,
            batch_size=1,
            original_log_odds=ctx.original_log_odds,
        )[0]
        eval_count += 1
        trajectory.append({"evals": eval_count, "best_score": current_best_score})

    return RegionResult(
        region=current_region,
        score=final_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )


def _generate_lookahead_candidates(
    current_region: frozenset[int],
    image_graph: ImageGraph,
    used_segments: Set[int],
    lookahead_distance: int,
    desired_length: int,
) -> dict[frozenset[int], int]:
    """Generate all connected supersets up to lookahead_distance steps via BFS.

    Args:
        current_region: Frozenset of the currently built region.
        image_graph: Graph representation of image segments and their adjacencies.
        used_segments: Set of globally excluded or already used segments.
        lookahead_distance: Maximum depth for the BFS expansion.
        desired_length: Maximum allowed total size of the candidate region.

    Returns:
        Dict mapping candidate_region -> first_step_segment_id
    """
    candidates: dict[frozenset[int], int] = {}  # region -> first_step

    # Queue stores tuples of: (current_region, first_step_that_led_here, current_depth)
    queue: deque[tuple[frozenset[int], int | None, int]] = deque(
        [(current_region, None, 0)]
    )
    visited = {current_region}

    while queue:
        region, first_step, depth = queue.popleft()

        # Store valid candidates (depth > 0)
        if depth > 0 and first_step is not None and region not in candidates:
            # Only add if not already seen (shortest path wins)
            candidates[region] = first_step

        # Stop expanding if we reached the lookahead limit or maximum size
        if depth >= lookahead_distance or len(region) >= desired_length:
            continue

        frontier = image_graph.get_frontier(region, used_segments)
        for seg_id in frontier:
            new_region = region | {seg_id}

            if new_region not in visited:
                visited.add(new_region)
                # If at the first layer (depth 0), this seg_id is our first_step.
                # Otherwise, pass along the first_step inherited from the parent.
                next_first_step = seg_id if depth == 0 else first_step
                queue.append((new_region, next_first_step, depth + 1))

    return candidates
