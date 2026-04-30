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
    num_evals: int = 6400,
    num_rollouts: int = 64,
    exploration_c: float = 1.4,
    alpha: float = 0.0,
) -> ExplanationMethodFn:
    """Return a function that generates an MCTS-based region building strategy.

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
        ExplanationMethodFn: Method computing contextual importance via MCTS search.
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
        """Find the region via Monte Carlo Tree Search."""
        from ciao.algorithm.mcts import build_region_mcts

        return build_region_mcts(
            ctx=ctx,
            num_iterations=num_iterations,
            num_rollouts=num_rollouts,
            exploration_c=exploration_c,
            alpha=alpha,
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
