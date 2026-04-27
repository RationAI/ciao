"""Monte Carlo Tree Search for connected image regions.

Supports:
- Leaf parallelization: ``num_rollouts`` random rollouts per selected leaf
- Terminal caching to avoid re-evaluating visited states
- Mean+max backup for value estimation
- Local UCT normalization across a node's children

Selection details:
- ``Q(s,a) = alpha * max_value + (1 - alpha) * mean_value``
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
    alpha: float,
) -> MCTSNode:
    """Select child with highest UCT score using local normalization."""
    children = list(node.children.values())
    q_values = [alpha * c.max_value + (1.0 - alpha) * c.mean_value for c in children]
    min_q = min(q_values)
    max_q = max(q_values)
    log_parent_visits = math.log(node.visits)

    def uct_score(child_q: tuple[MCTSNode, float]) -> float:
        child, q_value = child_q
        if max_q > min_q:
            q_norm = (2.0 * (q_value - min_q) / (max_q - min_q)) - 1.0
        else:
            q_norm = 1.0
        explore = exploration_c * math.sqrt(log_parent_visits / child.visits)
        return q_norm + explore

    return max(zip(children, q_values, strict=True), key=uct_score)[0]


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
    unexpanded = sorted(seg_id for seg_id in frontier if seg_id not in node.children)

    if not unexpanded:
        return None

    seg_id = random.choice(unexpanded)
    child_region = node.region | frozenset({seg_id})

    child = MCTSNode(region=child_region, parent=node)
    node.children[seg_id] = child

    return child


def backup_path(path: list[MCTSNode], rewards: list[float]) -> None:
    """Backup multiple rewards along a single path.

    Each reward contributes one visit to every node on the path.
    """
    k = len(rewards)
    sum_rewards = sum(rewards)
    max_reward = max(rewards)
    for node in path:
        new_visits = node.visits + k
        node.mean_value = (node.mean_value * node.visits + sum_rewards) / new_visits
        node.visits = new_visits
        if max_reward > node.max_value:
            node.max_value = max_reward


def simulate_leaf(
    ctx: SearchContext,
    leaf: MCTSNode,
    num_rollouts: int,
) -> tuple[list[float], list[frozenset[int]], int]:
    """Run ``num_rollouts`` simulations from ``leaf`` and return rewards.

    Terminal leaves are deterministic, so a single GPU evaluation (or cached
    value) is reused for all ``num_rollouts``. Non-terminal leaves sample
    independent random rollouts and dedupe before GPU evaluation.

    Returns rewards (already signed), the rollout regions per reward, and the
    number of GPU evaluations performed.
    """
    leaf_is_terminal = is_terminal(
        leaf.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
    )

    if leaf_is_terminal:
        if leaf.visits > 0:
            reward = leaf.mean_value  # already signed in prior backup
            evals = 0
        else:
            raw = calculate_region_deltas(
                predictor=ctx.predictor,
                input_batch=ctx.input_batch,
                segments=ctx.image_graph.segments,
                replacement_image=ctx.replacement_image,
                segment_sets=[leaf.region],
                target_class_idx=ctx.target_class_idx,
                batch_size=ctx.batch_size,
            )
            reward = raw[0] * ctx.optimization_sign
            evals = 1
        rewards = [reward] * num_rollouts
        regions = [leaf.region] * num_rollouts
        return rewards, regions, evals

    rollout_regions = [
        ctx.image_graph.sample_connected_superset(
            base_region=leaf.region,
            target_length=ctx.desired_length,
            used_segments=ctx.used_segments,
        )
        for _ in range(num_rollouts)
    ]
    unique_regions = list(dict.fromkeys(rollout_regions))
    raw_rewards = calculate_region_deltas(
        predictor=ctx.predictor,
        input_batch=ctx.input_batch,
        segments=ctx.image_graph.segments,
        replacement_image=ctx.replacement_image,
        segment_sets=unique_regions,
        target_class_idx=ctx.target_class_idx,
        batch_size=ctx.batch_size,
    )
    region_to_reward = {
        region: reward * ctx.optimization_sign
        for region, reward in zip(unique_regions, raw_rewards, strict=True)
    }
    rewards = [region_to_reward[region] for region in rollout_regions]
    return rewards, rollout_regions, len(unique_regions)


def build_region_mcts(
    ctx: SearchContext,
    num_iterations: int,
    num_rollouts: int,
    exploration_c: float,
    alpha: float,
) -> RegionResult:
    """Build a region using Monte Carlo Tree Search with leaf parallelization.

    Each iteration:
    1. Selection via UCT down to a leaf.
    2. Expansion of one new child (if non-terminal).
    3. ``num_rollouts`` independent random rollouts from the leaf.
    4. Backup of all rollout rewards along the path.

    Args:
        ctx: Search context with model state and parameters.
        num_iterations: Number of MCTS iterations.
        num_rollouts: Number of random rollouts per selected leaf.
        exploration_c: UCT exploration constant.
        alpha: Weight on max vs mean in the UCT Q-value,
            ``Q = alpha * max + (1 - alpha) * mean``. Must be in [0, 1].

    Returns:
        RegionResult with region, score, and stats.
    """
    root_region = frozenset({ctx.seed_idx})
    root = MCTSNode(region=root_region, parent=None)

    best_region = root.region
    best_score = -float("inf")

    eval_count = 0
    trajectory: list[dict[str, float]] = []

    for _ in tqdm(range(num_iterations), desc="MCTS", unit="iter"):
        # --- SELECTION + EXPANSION ---
        node = root
        path = [node]
        while True:
            if is_terminal(
                node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
            ):
                break
            if not is_fully_expanded(node, ctx.image_graph, ctx.used_segments):
                child = expand_node(node, ctx.image_graph, ctx.used_segments)
                assert child is not None  # selection invariant: not fully expanded
                node = child
                path.append(node)
                break
            node = select_uct_child(node, exploration_c, alpha)
            path.append(node)

        # --- SIMULATION ---
        rewards, rollout_regions, evals = simulate_leaf(ctx, node, num_rollouts)
        eval_count += evals

        for region, reward in zip(rollout_regions, rewards, strict=True):
            if reward > best_score:
                best_score = reward
                best_region = region

        trajectory.append({"evals": eval_count, "best_score": best_score})

        # --- BACKUP ---
        backup_path(path, rewards)

    best_score = best_score * ctx.optimization_sign

    return RegionResult(
        region=best_region,
        score=best_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )
