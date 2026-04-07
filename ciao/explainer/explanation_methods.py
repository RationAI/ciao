"""Base interfaces and implementations for explanation methods."""

from collections.abc import Callable

from ciao.algorithm.context import SearchContext
from ciao.scoring.hyperpixel import HyperpixelResult


ExplanationMethodFn = Callable[[SearchContext], HyperpixelResult]


def make_lookahead_method(lookahead_distance: int = 2) -> ExplanationMethodFn:
    """Return a function that generates a lookahead hyperpixel building strategy.

    Args:
        lookahead_distance: How many search context steps to look ahead during search.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via search algorithms.
    """
    if lookahead_distance < 1:
        raise ValueError(f"lookahead_distance must be >= 1, got {lookahead_distance}")

    def method(ctx: SearchContext) -> HyperpixelResult:
        """Find the hyperpixel via greedy exploration and distance lookahead."""
        from ciao.algorithm.lookahead import build_hyperpixel_greedy_lookahead

        return build_hyperpixel_greedy_lookahead(
            ctx=ctx,
            lookahead_distance=lookahead_distance,
        )

    return method
