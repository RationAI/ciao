"""UCB algorithm — Asynchronous Batched Upper Confidence Bound for region growth.

At each commit step we treat the frontier neighbors as bandit arms and pull them
with a UCB1-style policy. Rollouts are gathered into GPU-friendly batches; within
a batch we use virtual counts to avoid greedily picking the same arm. The arm
score blends mean and max reward to retain the optimization-extremum signal:

    score = alpha * max_signed + (1 - alpha) * mean_signed
          + c * sqrt(ln(N) / (count + virtual_count))

After the step's rollout budget is exhausted we commit the neighbor with the
highest real ``count`` (the one UCB explored the most), then repeat.
"""

import math
import time
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from ciao.algorithm.context import SearchContext
from ciao.algorithm.search_helpers import evaluate_and_cache
from ciao.scoring.region import RegionResult, calculate_region_deltas


@dataclass
class _ArmStats:
    """Per-neighbor running statistics for one commit step."""

    count: int = 0
    active_count: int = 0
    sum_signed: float = 0.0
    max_signed: float = -float("inf")
    virtual_count: int = 0
    extended_region: frozenset[int] = field(default_factory=frozenset)

    def mean(self) -> float:
        return self.sum_signed / self.count if self.count > 0 else 0.0

    def update(self, signed_score: float) -> None:
        self.count += 1
        self.sum_signed += signed_score
        self.max_signed = max(self.max_signed, signed_score)


@dataclass
class _SearchState:
    """Mutable bookkeeping shared across commit steps and batches.

    Pulling this out of the main function lets ``_ucb_step`` log to the
    trajectory after each batch evaluation (granular) instead of only once
    per commit step.
    """

    t0: float
    eval_count: int = 0
    best_signed: float = -float("inf")
    best_region: frozenset[int] = field(default_factory=frozenset)
    best_raw_score: float = 0.0
    trajectory: list[dict[str, float]] = field(default_factory=list)

    def observe(self, region: frozenset[int], raw_score: float, signed: float) -> None:
        if signed > self.best_signed:
            self.best_signed = signed
            self.best_region = region
            self.best_raw_score = raw_score

    def log(self) -> None:
        self.trajectory.append(
            {
                "evals": self.eval_count,
                "best_score": self.best_raw_score,
                "time": time.monotonic() - self.t0,
            }
        )


def build_region_ucb(
    ctx: SearchContext,
    step_budget: int,
    batch_size: int,
    ucb_c: float = 1.0,
    ucb_alpha: float = 0.5,
) -> RegionResult:
    """Build a region by stepwise asynchronous-batched UCB over the frontier.

    Args:
        ctx: Search context.
        step_budget: Maximum number of rollouts per commit step. The first
            ``len(frontier)`` rollouts seed each neighbor; remaining rollouts
            are scheduled by UCB.
        batch_size: Number of rollouts gathered before each GPU evaluation pass.
        ucb_c: Exploration constant for the UCB1 bonus term.
        ucb_alpha: Blend weight between max and mean for the exploitation
            term (``alpha * max + (1 - alpha) * mean``). Default 0.5.

    Returns:
        RegionResult with the best-scoring region encountered during the search.
    """
    if step_budget < 1:
        raise ValueError(f"step_budget must be >= 1, got {step_budget}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not 0.0 <= ucb_alpha <= 1.0:
        raise ValueError(f"ucb_alpha must be in [0, 1], got {ucb_alpha}")

    used_segments = set(ctx.used_segments)
    curr_region = frozenset({ctx.seed_idx})

    evaluated_scores: dict[frozenset[int], float] = {}

    state = _SearchState(t0=time.monotonic(), best_region=curr_region)

    while len(curr_region) < ctx.desired_length:
        current_frontier = ctx.image_graph.get_frontier(curr_region, used_segments)
        if not current_frontier:
            break

        arms = _ucb_step(
            curr_region=curr_region,
            current_frontier=current_frontier,
            step_budget=step_budget,
            batch_size=batch_size,
            ucb_c=ucb_c,
            ucb_alpha=ucb_alpha,
            ctx=ctx,
            used_segments=used_segments,
            evaluated_scores=evaluated_scores,
            state=state,
        )

        valid_arms = [n for n, s in arms.items() if s.count > 0]
        if not valid_arms:
            break

        winner = max(valid_arms, key=lambda n: arms[n].active_count)
        curr_region = curr_region | {winner}

    if state.best_signed == -float("inf"):
        score = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=[curr_region],
            target_class_idx=ctx.target_class_idx,
            original_log_odds=ctx.original_log_odds,
            batch_size=ctx.batch_size,
        )[0]
        state.eval_count += 1
        state.observe(curr_region, score, score * ctx.optimization_sign)
        state.log()

    return RegionResult(
        region=state.best_region,
        score=state.best_raw_score,
        evaluations_count=state.eval_count,
        trajectory=state.trajectory,
    )


def _ucb_score(
    arm: _ArmStats, total_pulls: int, ucb_c: float, ucb_alpha: float
) -> float:
    """Blended-mean UCB1 score for a single arm.

    Unvisited arms get +inf so the first ``len(frontier)`` rollouts seed every arm.
    """
    if arm.active_count == 0:
        return float("inf")
    exploit = ucb_alpha * arm.max_signed + (1.0 - ucb_alpha) * arm.mean()
    explore = ucb_c * math.sqrt(
        math.log(max(total_pulls, 1)) / (arm.count + arm.virtual_count)
    )
    return exploit + explore


def _ucb_step(
    curr_region: AbstractSet[int],
    current_frontier: AbstractSet[int],
    step_budget: int,
    batch_size: int,
    ucb_c: float,
    ucb_alpha: float,
    ctx: SearchContext,
    used_segments: AbstractSet[int],
    evaluated_scores: dict[frozenset[int], float],
    state: _SearchState,
) -> dict[int, _ArmStats]:
    """Run one commit step of asynchronous-batched UCB.

    Mutates ``state`` (eval_count, best-region, trajectory) once per batch so
    the trajectory has granular entries within a single commit step.

    Returns the per-arm stats so the caller can pick the commit winner.
    """
    arms: dict[int, _ArmStats] = {
        n: _ArmStats(extended_region=frozenset(curr_region | {n}))
        for n in current_frontier
    }

    rollouts_done = 0

    while rollouts_done < step_budget:
        # --- Fill one batch ---
        this_batch = min(batch_size, step_budget - rollouts_done)
        batch_regions: list[frozenset[int]] = []
        batch_owners: list[int] = []  # arm chosen by UCB for each rollout

        for arm in arms.values():
            arm.virtual_count = 0

        for _ in range(this_batch):
            chosen = max(
                arms.keys(),
                key=lambda n: _ucb_score(arms[n], rollouts_done, ucb_c, ucb_alpha),
            )
            arm = arms[chosen]
            sampled = ctx.image_graph.sample_connected_superset(
                base_region=arm.extended_region,
                target_length=ctx.desired_length,
                used_segments=used_segments,
            )
            batch_regions.append(sampled)
            batch_owners.append(chosen)
            arm.virtual_count += 1
            arm.active_count += 1
            rollouts_done += 1

        # --- Evaluate uncached regions ---
        regions_to_evaluate = [
            r for r in set(batch_regions) if r not in evaluated_scores
        ]
        state.eval_count += evaluate_and_cache(
            regions_to_evaluate, evaluated_scores, ctx
        )

        # --- Update real stats with bucketization ---
        # Each rollout reward is credited to (a) the arm that chose it, and
        # (b) every other frontier neighbor that happens to be in the rollout.
        for region, owner in zip(batch_regions, batch_owners, strict=True):
            score = evaluated_scores[region]
            signed = score * ctx.optimization_sign
            state.observe(region, score, signed)

            arms[owner].update(signed)
            for hit in (region & current_frontier) - {owner}:
                arms[hit].update(signed)

        for arm in arms.values():
            arm.virtual_count = 0

        # Granular trajectory: log after each batch evaluation.
        state.log()

    return arms
