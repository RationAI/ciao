"""Pure Monte Carlo search for connected image regions."""

import time

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult, calculate_region_deltas


def build_region_pure_monte_carlo(
    ctx: SearchContext,
    num_evals: int,
    patience: int | None = None,
) -> RegionResult:
    """Build a region via random sampling from the seed and pick the best sample.

    Samples connected supersets one at a time, deduplicating against previously
    sampled regions. Once ``ctx.batch_size`` new uniques have accumulated, they
    are scored together and a trajectory point is recorded. Sampling stops when
    ``num_evals`` unique regions have been scored, or when ``patience``
    consecutive samples yield only duplicates.

    Args:
        ctx: Search context with model state and search parameters.
        num_evals: Target number of unique regions to score.
        patience: Stop early after this many consecutive duplicate samples.
            Defaults to ``num_evals`` (effectively disabled).

    Returns:
        RegionResult containing the best sampled region and its score.
    """
    if num_evals < 1:
        raise ValueError(f"num_evals must be >= 1, got {num_evals}")
    if patience is None:
        patience = num_evals
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    seed_region = frozenset({ctx.seed_idx})
    batch_size = ctx.batch_size

    t0 = time.monotonic()
    seen: dict[frozenset[int], None] = {}
    pending: list[frozenset[int]] = []
    scores: list[float] = []
    signed_scores: list[float] = []
    trajectory: list[dict[str, float]] = []
    best_signed = -float("inf")
    consecutive_duplicates = 0

    def flush_pending() -> None:
        nonlocal best_signed
        if not pending:
            return
        batch_scores = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=pending,
            target_class_idx=ctx.target_class_idx,
            original_log_odds=ctx.original_log_odds,
            batch_size=batch_size,
        )
        batch_signed = [s * ctx.optimization_sign for s in batch_scores]
        scores.extend(batch_scores)
        signed_scores.extend(batch_signed)
        best_signed = max(best_signed, max(batch_signed))
        trajectory.append(
            {
                "evals": len(scores),
                "best_score": best_signed * ctx.optimization_sign,
                "time": time.monotonic() - t0,
            }
        )
        pending.clear()

    while len(seen) < num_evals and consecutive_duplicates < patience:
        region = ctx.image_graph.sample_connected_superset(
            base_region=seed_region,
            target_length=ctx.desired_length,
            used_segments=ctx.used_segments,
        )
        if region in seen:
            consecutive_duplicates += 1
            continue
        seen[region] = None
        pending.append(region)
        consecutive_duplicates = 0
        if len(pending) >= batch_size:
            flush_pending()

    # Score any leftover uniques that didn't fill a full batch before the loop exited.
    flush_pending()

    unique_regions = list(seen.keys())
    best_idx = max(range(len(signed_scores)), key=lambda i: signed_scores[i])

    return RegionResult(
        region=unique_regions[best_idx],
        score=scores[best_idx],
        evaluations_count=len(scores),
        trajectory=trajectory,
    )
