"""Beam search for connected image regions using precomputed segment scores.

The search objective uses the initial segment scores only. No neural-network queries
are made during expansion; the selected final region is evaluated once at the end.
"""

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_beam_search(
    ctx: SearchContext,
    beam_width: int,
) -> RegionResult:
    """Build a region with beam search over connected segment expansions.

    Args:
        ctx: Search context with graph state and precomputed segment scores.
        beam_width: Number of best partial regions kept per depth.

    Returns:
        RegionResult containing the selected region and one final NN-evaluated score.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    target_length = ctx.desired_length
    used_segments = set(ctx.used_segments)

    seed_region = frozenset({ctx.seed_idx})
    seed_signed_sum = ctx.optimization_sign * ctx.segment_scores[ctx.seed_idx]

    # Beam stores tuples of (region, signed_sum_of_initial_segment_scores)
    beam: list[tuple[frozenset[int], float]] = [(seed_region, seed_signed_sum)]

    while beam and len(beam[0][0]) < target_length:
        next_level: dict[frozenset[int], float] = {}

        for region, signed_sum in beam:
            frontier = ctx.image_graph.get_frontier(region, used_segments)
            for seg_id in sorted(frontier):
                if seg_id not in ctx.segment_scores:
                    raise ValueError(f"segment {seg_id} is missing from segment_scores")

                new_region = region | {seg_id}
                new_signed_sum = (
                    signed_sum + ctx.optimization_sign * ctx.segment_scores[seg_id]
                )

                prev_best = next_level.get(new_region)
                if prev_best is None or new_signed_sum > prev_best:
                    next_level[new_region] = new_signed_sum

        if not next_level:
            break

        beam = sorted(next_level.items(), key=lambda item: item[1], reverse=True)[
            :beam_width
        ]

    full_length_candidates = [item for item in beam if len(item[0]) == target_length]
    if full_length_candidates:
        best_region = max(full_length_candidates, key=lambda item: item[1])[0]
    else:
        # Fallback when no region can reach target_length due graph constraints.
        best_region = max(beam, key=lambda item: (len(item[0]), item[1]))[0]

    final_score = calculate_region_deltas(
        predictor=ctx.predictor,
        input_batch=ctx.input_batch,
        segments=ctx.image_graph.segments,
        replacement_image=ctx.replacement_image,
        segment_sets=[best_region],
        target_class_idx=ctx.target_class_idx,
        batch_size=1,
    )[0]

    return RegionResult(region=best_region, score=final_score)
