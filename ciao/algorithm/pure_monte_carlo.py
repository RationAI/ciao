"""Pure Monte Carlo search for connected image regions."""

import time
from collections.abc import Iterator
from itertools import batched

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult, calculate_region_deltas


def _sample_unique_regions(
    ctx: SearchContext, num_evals: int, patience: int
) -> Iterator[frozenset[int]]:
    """Yield distinct connected supersets until the budget or patience runs out."""
    seed_region = frozenset({ctx.seed_idx})
    seen: set[frozenset[int]] = set()
    consecutive_duplicates = 0

    while len(seen) < num_evals and consecutive_duplicates < patience:
        region = ctx.image_graph.sample_connected_superset(
            base_region=seed_region,
            target_length=ctx.desired_length,
            used_segments=ctx.used_segments,
        )
        if region in seen:
            consecutive_duplicates += 1
            continue
        seen.add(region)
        consecutive_duplicates = 0
        yield region


def _best(
    scored: list[tuple[frozenset[int], float]], sign: int
) -> tuple[frozenset[int], float]:
    return max(scored, key=lambda region_score: region_score[1] * sign)


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

    t0 = time.monotonic()
    scored: list[tuple[frozenset[int], float]] = []
    trajectory: list[dict[str, float]] = []

    for batch in batched(
        _sample_unique_regions(ctx, num_evals, patience), ctx.batch_size
    ):
        batch_scores = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=list(batch),
            target_class_idx=ctx.target_class_idx,
            original_log_odds=ctx.original_log_odds,
            batch_size=ctx.batch_size,
        )
        scored.extend(zip(batch, batch_scores, strict=True))
        trajectory.append(
            {
                "evals": len(scored),
                "best_score": _best(scored, ctx.optimization_sign)[1],
                "time": time.monotonic() - t0,
            }
        )

    best_region, best_score = _best(scored, ctx.optimization_sign)

    return RegionResult(
        region=best_region,
        score=best_score,
        evaluations_count=len(scored),
        trajectory=trajectory,
    )
