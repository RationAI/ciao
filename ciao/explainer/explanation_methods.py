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


def make_potential_method(num_simulations: int = 10) -> ExplanationMethodFn:
    """Return a function that generates a potential-based region building strategy.

    Args:
        num_simulations: Number of Monte Carlo simulations per frontier node.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via potential search.
    """
    if num_simulations < 1:
        raise ValueError(f"num_simulations must be >= 1, got {num_simulations}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via sequential Monte Carlo with potential-based selection."""
        from ciao.algorithm.potential import build_region_potential

        return build_region_potential(
            ctx=ctx,
            num_simulations=num_simulations,
        )

    return method
