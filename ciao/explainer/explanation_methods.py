"""Base interfaces and implementations for explanation methods."""

from ciao.algorithm.context import SearchContext
from ciao.scoring.region import RegionResult
from ciao.typing import ExplanationMethodFn


def make_lookahead_method(lookahead_distance: int = 2) -> ExplanationMethodFn:
    """Return a function that generates a lookahead region building strategy.

    Args:
        lookahead_distance: How many search context steps to look ahead during search.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via search algorithms.
    """
    if lookahead_distance < 1:
        raise ValueError(f"lookahead_distance must be >= 1, got {lookahead_distance}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via greedy exploration and distance lookahead."""
        from ciao.algorithm.lookahead import build_region_greedy_lookahead

        return build_region_greedy_lookahead(
            ctx=ctx,
            lookahead_distance=lookahead_distance,
        )

    return method


def make_pure_monte_carlo_method(
    num_evals: int = 100,
    patience: int | None = None,
) -> ExplanationMethodFn:
    """Return a function that generates a pure Monte-Carlo region strategy.

    Args:
        num_evals: Target number of unique connected supersets to score.
        patience: Stop early after this many consecutive duplicate samples.
            Defaults to ``num_evals`` (effectively disabled).

    Returns:
        ExplanationMethodFn: Method computing contextual importance via pure sampling.
    """
    if num_evals < 1:
        raise ValueError(f"num_evals must be >= 1, got {num_evals}")
    if patience is not None and patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region by pure random sampling from the seed."""
        from ciao.algorithm.pure_monte_carlo import build_region_pure_monte_carlo

        return build_region_pure_monte_carlo(
            ctx=ctx,
            num_evals=num_evals,
            patience=patience,
        )

    return method
