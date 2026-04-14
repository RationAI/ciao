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


def make_beam_search_method(beam_width: int = 64) -> ExplanationMethodFn:
    """Return a function that generates a beam-search region building strategy.

    Beam search expands connected regions using precomputed segment scores only,
    then performs one final NN pass for the selected region.

    Args:
        beam_width: Number of best partial regions kept per depth.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via beam search.
    """
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via score-only beam search."""
        from ciao.algorithm.beam_search_precomputed import build_region_beam_search

        return build_region_beam_search(
            ctx=ctx,
            beam_width=beam_width,
        )

    return method
