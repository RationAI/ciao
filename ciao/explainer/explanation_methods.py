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


def make_mcts_method(
    num_iterations: int = 100,
    exploration_c: float = 1.4,
    virtual_loss: float = 1.0,
) -> ExplanationMethodFn:
    """Return a function that generates an MCTS-based region building strategy.

    Args:
        num_iterations: Number of MCTS iterations (batch collections).
        exploration_c: UCT exploration constant.
        virtual_loss: Multiplier for pending counter in UCT.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via MCTS search.
    """
    if num_iterations < 1:
        raise ValueError(f"num_iterations must be >= 1, got {num_iterations}")
    if exploration_c <= 0:
        raise ValueError()
    if virtual_loss <= 0:
        raise ValueError()

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via Monte Carlo Tree Search."""
        from ciao.algorithm.mcts import build_region_mcts

        return build_region_mcts(
            ctx=ctx,
            num_iterations=num_iterations,
            exploration_c=exploration_c,
            virtual_loss=virtual_loss,
        )

    return method
