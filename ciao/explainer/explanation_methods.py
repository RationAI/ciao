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


def make_mcgs_method(
    num_evals: int = 6400,
    num_rollouts: int = 64,
    exploration_c: float = 1.4,
    alpha: float = 0.0,
) -> ExplanationMethodFn:
    """Return a function that generates an MCGS-based region building strategy.

    Args:
        num_evals: Approximate budget on GPU evaluations. The actual number of
            iterations is ``num_evals // num_rollouts``. Cached terminal
            revisits do not consume the budget, so realized eval count may be
            lower than ``num_evals``.
        num_rollouts: Number of random rollouts per selected leaf (leaf parallelization).
        exploration_c: UCT exploration constant.
        alpha: Weight on max vs mean in the UCT Q-value,
            ``Q = alpha * max + (1 - alpha) * mean``. Must be in [0, 1].

    Returns:
        ExplanationMethodFn: Method computing contextual importance via MCGS search.
    """
    if num_rollouts < 1:
        raise ValueError(f"num_rollouts must be >= 1, got {num_rollouts}")
    if num_evals < num_rollouts:
        raise ValueError(
            f"num_evals must be >= num_rollouts ({num_rollouts}), got {num_evals}"
        )
    if exploration_c <= 0:
        raise ValueError(f"exploration_c must be > 0, got {exploration_c}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    num_iterations = num_evals // num_rollouts

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via Monte Carlo Graph Search."""
        from ciao.algorithm.mcgs import build_region_mcgs

        return build_region_mcgs(
            ctx=ctx,
            num_iterations=num_iterations,
            num_rollouts=num_rollouts,
            exploration_c=exploration_c,
            alpha=alpha,
        )

    return method
