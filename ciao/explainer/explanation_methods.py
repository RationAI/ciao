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


def make_potential_method(step_budget: int = 10) -> ExplanationMethodFn:
    """Return a function that generates a potential-based region building strategy.

    Args:
        step_budget: Total number of rollouts per commit step, distributed
            round-robin across frontier nodes.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via potential search.
    """
    if step_budget < 1:
        raise ValueError(f"step_budget must be >= 1, got {step_budget}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via sequential Monte Carlo with potential-based selection."""
        from ciao.algorithm.potential import build_region_potential

        return build_region_potential(
            ctx=ctx,
            step_budget=step_budget,
        )

    return method


def make_ucb_method(
    step_budget: int = 64,
    batch_size: int = 16,
    ucb_c: float = 1.0,
    ucb_alpha: float = 0.5,
) -> ExplanationMethodFn:
    """Return a function that builds regions via asynchronous-batched UCB.

    Args:
        step_budget: Total rollouts per commit step.
        batch_size: Rollouts gathered per GPU evaluation pass.
        ucb_c: Exploration constant for the UCB1 bonus term.
        ucb_alpha: Blend weight ``alpha * max + (1 - alpha) * mean`` for the
            exploitation term.

    Returns:
        ExplanationMethodFn: Method computing contextual importance via UCB search.
    """
    if step_budget < 1:
        raise ValueError(f"step_budget must be >= 1, got {step_budget}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not 0.0 <= ucb_alpha <= 1.0:
        raise ValueError(f"ucb_alpha must be in [0, 1], got {ucb_alpha}")

    def method(ctx: SearchContext) -> RegionResult:
        """Find the region via asynchronous-batched UCB."""
        from ciao.algorithm.ucb import build_region_ucb

        return build_region_ucb(
            ctx=ctx,
            step_budget=step_budget,
            batch_size=batch_size,
            ucb_c=ucb_c,
            ucb_alpha=ucb_alpha,
        )

    return method
