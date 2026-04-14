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


def make_pure_monte_carlo_method(num_simulations: int = 100) -> ExplanationMethodFn:
    """Return a function that generates a pure Monte-Carlo region strategy.

    Args:
        num_simulations: Number of connected supersets sampled from the seed.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via pure sampling.
    """
    if num_simulations < 1:
        raise ValueError(f"num_simulations must be >= 1, got {num_simulations}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region by pure random sampling from the seed."""
        from ciao.algorithm.pure_monte_carlo import build_region_pure_monte_carlo

        return build_region_pure_monte_carlo(
            ctx=ctx,
            num_simulations=num_simulations,
        )

    return method
