"""Monte Carlo Tree Search for connected image regions.

Supports:
- Batch collection and evaluation for GPU efficiency
- Virtual loss for parallel safety (via ``pending`` counters)
- Terminal caching to avoid re-evaluating visited states
- Mean backup for value estimation
- Local UCT normalization across a node's children

Selection details:
- ``Q(s,a) = alpha * max_value + (1 - alpha) * mean_value``
- ``Q_eff(s,a) = Q(s,a) - w * V_L * |Q(s,a)|``
- ``Q_norm`` is local min-max normalization to ``[-1, 1]``
- ``UCT(s,a) = Q_norm + C * sqrt(ln(N_parent) / n_j)``

State is represented as ``frozenset[int]`` of selected segment IDs.
"""

import math
import random

from tqdm import tqdm

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.algorithm.nodes import MCTSNode
from ciao.algorithm.search_helpers import is_terminal
from ciao.scoring.region import RegionResult, calculate_region_deltas


def is_fully_expanded(
    node: MCTSNode, image_graph: ImageGraph, used_region: frozenset[int]
) -> bool:
    """Check if all frontier segments have been expanded as children."""
    frontier = image_graph.get_frontier(node.region, used_region)
    return frontier.issubset(node.children)


def select_uct_child(
    node: MCTSNode,
    exploration_c: float,
    virtual_loss: float,
    alpha: float,
) -> MCTSNode | None:
    """Select child with highest UCT score using local normalization and FPU."""
    children = list(node.children.values())
    if not children:
        return None

    # for FPU
    parent_q = (
        alpha * node.max_value + (1.0 - alpha) * node.mean_value
        if node.visits > 0
        else 0.0
    )

    q_eff_values: list[float] = []
    for child in children:
        # FPU
        if child.visits > 0:
            q_value = alpha * child.max_value + (1.0 - alpha) * child.mean_value
        else:
            q_value = parent_q

        # Apply virtual loss through the pending counter.
        q_eff = q_value - virtual_loss * child.pending * abs(q_value)
        q_eff_values.append(q_eff)

    parent_n = node.visits + node.pending

    min_q_eff = min(q_eff_values)
    max_q_eff = max(q_eff_values)

    best_uct = -float("inf")
    best_child = None

    for child, q_eff in zip(children, q_eff_values, strict=True):
        if max_q_eff > min_q_eff:
            q_norm = (2.0 * (q_eff - min_q_eff) / (max_q_eff - min_q_eff)) - 1.0
        else:
            q_norm = 1.0

        child_n = child.visits + child.pending
        explore = exploration_c * math.sqrt(math.log(parent_n) / child_n)
        score = q_norm + explore

        if score > best_uct:
            best_uct = score
            best_child = child

    return best_child


def expand_node(
    node: MCTSNode,
    image_graph: ImageGraph,
    used_region: frozenset[int],
) -> MCTSNode | None:
    """Standard expansion: Pick one random unexpanded segment.

    Args:
        node: Node to expand
        image_graph: ImageGraph to compute adjacencies
        used_region: Globally excluded segments

    Returns:
        New child node if created, None if no expansion possible
    """
    frontier = image_graph.get_frontier(node.region, used_region)
    unexpanded = [seg_id for seg_id in frontier if seg_id not in node.children]

    if not unexpanded:
        return None

    # Create one new child
    seg_id = random.choice(unexpanded)
    child_region = node.region | frozenset({seg_id})

    child = MCTSNode(region=child_region, parent=node)
    node.children[seg_id] = child

    return child


def backup_paths(batch_paths: list[list[MCTSNode]], rewards: list[float]) -> None:
    """Backup rewards using standard statistics.

    Updates:
    - visits
    - mean_value (mean backup for selection)
    - max_value (best reward seen through this node)
    - pending (release virtual loss)
    """
    for path, reward in zip(batch_paths, rewards, strict=True):
        for node in path:
            if node.pending > 0:
                node.pending -= 1  # Release virtual loss where it was applied
            node.visits += 1
            node.mean_value += (reward - node.mean_value) / node.visits
            if reward > node.max_value:
                node.max_value = reward


def _collect_mcts_batch(
    ctx: SearchContext,
    root: MCTSNode,
    exploration_c: float,
    virtual_loss: float,
    alpha: float,
) -> tuple[list[list[MCTSNode]], list[frozenset[int]], list[float | None]]:
    """Phase 1: Batch Collection - Selection, Expansion, and Simulation."""
    batch_paths: list[list[MCTSNode]] = []
    batch_rollout_regions: list[frozenset[int]] = []
    cached_rewards: list[float | None] = []  # Store cached values for visited terminals

    for _ in range(ctx.batch_size):
        # --- SELECTION ---
        node = root
        node.pending += 1  # Count in-flight rollout starting at the root.
        path = [node]

        # Standard selection using mean-value UCT
        while is_fully_expanded(
            node, ctx.image_graph, ctx.used_segments
        ) and not is_terminal(
            node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
        ):
            child = select_uct_child(node, exploration_c, virtual_loss, alpha)

            if child is None:
                raise RuntimeError(
                    "Selection failed to find a child, but node is fully expanded."
                )

            child.pending += 1
            node = child
            path.append(node)

        # --- EXPANSION ---
        if not is_terminal(
            node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
        ):
            child = expand_node(node, ctx.image_graph, ctx.used_segments)

            if child is not None:
                child.pending += 1
                node = child
                path.append(node)

        # --- SIMULATION (or cache lookup) ---
        # Check terminal cache
        if (
            is_terminal(
                node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
            )
            and node.visits > 0
        ):
            # Reuse cached value - no GPU evaluation needed
            rollout_region = node.region
            cached_rewards.append(node.mean_value)
        else:
            # Need GPU evaluation
            if is_terminal(
                node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
            ):
                rollout_region = node.region
            else:
                # Random rollout
                rollout_region = ctx.image_graph.sample_connected_superset(
                    base_region=node.region,
                    target_length=ctx.desired_length,
                    used_segments=ctx.used_segments,
                )

            cached_rewards.append(None)

        batch_paths.append(path)
        batch_rollout_regions.append(rollout_region)

    return batch_paths, batch_rollout_regions, cached_rewards


def _evaluate_mcts_batch(
    ctx: SearchContext,
    batch_rollout_regions: list[frozenset[int]],
    cached_rewards: list[float | None],
) -> tuple[list[float], int]:
    """Phase 2: Batch Evaluation - dedupe uncached regions and merge rewards.

    Returns the per-path rewards and the number of unique GPU evaluations performed.
    """
    uncached_regions = [
        batch_rollout_regions[i]
        for i, reward in enumerate(cached_rewards)
        if reward is None
    ]
    unique_regions = list(dict.fromkeys(uncached_regions))

    region_rewards: dict[frozenset[int], float] = {}
    if unique_regions:
        raw_rewards = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=unique_regions,
            target_class_idx=ctx.target_class_idx,
            batch_size=ctx.batch_size,
        )
        region_rewards = {
            region: reward * ctx.optimization_sign
            for region, reward in zip(unique_regions, raw_rewards, strict=True)
        }

    # Merge GPU results with cached values (cached values are already signed)
    batch_rewards: list[float] = []
    for idx, cached_val in enumerate(cached_rewards):
        if cached_val is not None:
            batch_rewards.append(cached_val)
        else:
            batch_rewards.append(region_rewards[batch_rollout_regions[idx]])

    return batch_rewards, len(unique_regions)


def build_region_mcts(
    ctx: SearchContext,
    num_iterations: int,
    exploration_c: float = 1.4,
    virtual_loss: float = 1.0,
    alpha: float = 0.0,
) -> RegionResult:
    """Build a region using Monte Carlo Tree Search.

    Includes:
    - Batch collection and evaluation
    - Terminal caching to avoid re-evaluation
    - Virtual loss for parallel safety
    - Mean+max backup for value estimation

    Args:
        ctx: Search context with model state and parameters
        num_iterations: Number of MCTS iterations (batch collections)
        exploration_c: UCT exploration constant
        virtual_loss: Multiplier for pending counter in UCT
        alpha: Weight on max vs mean in the UCT Q-value,
            ``Q = alpha * max + (1 - alpha) * mean``. Must be in [0, 1].

    Returns:
        RegionResult with region, score, and stats.
    """
    # Create root node
    root_region = frozenset({ctx.seed_idx})
    root = MCTSNode(region=root_region, parent=None)

    best_region = root.region
    best_score = -float("inf")

    eval_count = 0
    trajectory: list[dict[str, float]] = []

    # --- MAIN MCTS LOOP ---
    for _ in tqdm(range(num_iterations), desc="MCTS", unit="iter"):
        # --- PHASE 1: BATCH COLLECTION ---
        batch_paths, batch_rollout_regions, cached_rewards = _collect_mcts_batch(
            ctx=ctx,
            root=root,
            exploration_c=exploration_c,
            virtual_loss=virtual_loss,
            alpha=alpha,
        )

        # --- PHASE 2: BATCH EVALUATION ---
        batch_rewards, batch_eval_count = _evaluate_mcts_batch(
            ctx=ctx,
            batch_rollout_regions=batch_rollout_regions,
            cached_rewards=cached_rewards,
        )
        eval_count += batch_eval_count

        # Update best region if we found a better one
        for i, reward in enumerate(batch_rewards):
            if reward > best_score:
                best_score = reward
                best_region = batch_rollout_regions[i]

        trajectory.append({"evals": eval_count, "best_score": best_score})

        # --- PHASE 3: BATCH BACKUP ---
        backup_paths(batch_paths, batch_rewards)

    best_score = best_score * ctx.optimization_sign

    return RegionResult(
        region=best_region,
        score=best_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )
