"""Monte Carlo Graph Search for connected image regions.

Differs from MCTS by sharing nodes for states reachable via multiple action
sequences: a transposition table maps ``frozenset[int]`` regions to nodes,
so the search forms a DAG instead of a tree. Edge statistics (per
``(parent, action)``) are tracked separately from node statistics, since
the same node can be reached via different parents.

Supports:
- Leaf parallelization: ``num_rollouts`` random rollouts per selected leaf
- Eager expansion with grafting from the transposition table
- Terminal caching to avoid re-evaluating visited states
- Mean+max backup at both node and edge level

Selection details (per outgoing edge of a fully-expanded node):
- ``Q(s,a) = alpha * edge.max_value + (1 - alpha) * edge.mean_value``
- ``Q_norm`` is local min-max normalization to ``[-1, 1]``
- ``UCT(s,a) = Q_norm + C * sqrt(ln(N_parent) / edge.visits)``
"""

import math
import random

from tqdm import tqdm

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.algorithm.nodes import EdgeStats, MCGSNode
from ciao.algorithm.search_helpers import is_terminal
from ciao.scoring.region import RegionResult, calculate_region_deltas


def select_uct_child(
    node: MCGSNode,
    exploration_c: float,
    alpha: float,
) -> tuple[int, MCGSNode]:
    """Select edge (action, child) with highest UCT score using edge statistics."""
    actions = list(node.children.keys())

    # Prefer edges that have never been traversed from this parent. With grafting,
    # a child node may have visits > 0 (visited via a different parent) while the
    # edge from this parent has edge.visits == 0.
    unvisited = [a for a in actions if node.edge_stats[a].visits == 0]
    if unvisited:
        action = random.choice(unvisited)
        return action, node.children[action]

    q_values = [
        alpha * node.edge_stats[a].max_value
        + (1.0 - alpha) * node.edge_stats[a].mean_value
        for a in actions
    ]
    min_q = min(q_values)
    max_q = max(q_values)
    total_edge_visits = sum(node.edge_stats[a].visits for a in actions)
    log_parent_visits = math.log(max(1, total_edge_visits))

    def uct_score(action_q: tuple[int, float]) -> float:
        action, q_value = action_q
        if max_q > min_q:
            q_norm = (2.0 * (q_value - min_q) / (max_q - min_q)) - 1.0
        else:
            q_norm = 1.0
        edge_n = node.edge_stats[action].visits
        explore = exploration_c * math.sqrt(log_parent_visits / edge_n)
        return q_norm + explore

    best_action = max(zip(actions, q_values, strict=True), key=uct_score)[0]
    return best_action, node.children[best_action]


def expand_node_eager(
    node: MCGSNode,
    image_graph: ImageGraph,
    used_region: frozenset[int],
    transposition_table: dict[frozenset[int], MCGSNode],
) -> tuple[int, MCGSNode] | None:
    """Eager expansion with grafting.

    For each frontier segment not yet a child:
    - If the resulting region exists in the transposition table, graft it
      (link the existing node as a child of ``node``).
    - Otherwise, mark as a fresh-creation candidate.

    One random fresh candidate is then created and returned. Returns ``None``
    if every frontier segment was already a child or could be grafted.
    """
    frontier = image_graph.get_frontier(node.region, used_region)

    new_candidates: list[tuple[int, frozenset[int]]] = []

    for seg_id in sorted(frontier):
        if seg_id in node.children:
            continue

        new_region = node.region | frozenset({seg_id})
        if new_region in transposition_table:
            existing = transposition_table[new_region]
            node.children[seg_id] = existing
            node.edge_stats[seg_id] = EdgeStats()
        else:
            new_candidates.append((seg_id, new_region))

    if not new_candidates:
        return None

    seg_id, new_region = random.choice(new_candidates)
    child = MCGSNode(region=new_region)
    transposition_table[new_region] = child
    node.children[seg_id] = child
    node.edge_stats[seg_id] = EdgeStats()

    return seg_id, child


def backup_path(
    path: list[MCGSNode],
    actions: list[int],
    rewards: list[float],
) -> None:
    """Backup multiple rewards along a single path: nodes and edges.

    Each reward contributes one visit to every node on the path and to every
    edge taken. ``actions[i]`` is the action taken at ``path[i]`` to reach
    ``path[i+1]``.
    """
    k = len(rewards)
    sum_rewards = sum(rewards)
    max_reward = max(rewards)

    for i, node in enumerate(path):
        new_visits = node.visits + k
        node.mean_value = (node.mean_value * node.visits + sum_rewards) / new_visits
        node.visits = new_visits
        if max_reward > node.max_value:
            node.max_value = max_reward

        if i < len(actions):
            edge = node.edge_stats[actions[i]]
            edge_new_visits = edge.visits + k
            edge.mean_value = (
                edge.mean_value * edge.visits + sum_rewards
            ) / edge_new_visits
            edge.visits = edge_new_visits
            if max_reward > edge.max_value:
                edge.max_value = max_reward


def simulate_leaf(
    ctx: SearchContext,
    leaf: MCGSNode,
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


def build_region_mcgs(
    ctx: SearchContext,
    num_iterations: int,
    num_rollouts: int,
    exploration_c: float,
    alpha: float,
) -> RegionResult:
    """Build a region using Monte Carlo Graph Search with leaf parallelization.

    Each iteration:
    1. Walk down the DAG, eager-expanding (with grafting) at each step;
       descend via UCT through fully-expanded nodes until either a freshly
       created node or a terminal is reached.
    2. ``num_rollouts`` independent random rollouts from the leaf.
    3. Backup of all rollout rewards along the path (nodes and edges).

    Args:
        ctx: Search context with model state and parameters.
        num_iterations: Number of MCGS iterations.
        num_rollouts: Number of random rollouts per selected leaf.
        exploration_c: UCT exploration constant.
        alpha: Weight on max vs mean in the UCT Q-value,
            ``Q = alpha * max + (1 - alpha) * mean``. Must be in [0, 1].

    Returns:
        RegionResult with region, score, and stats.
    """
    transposition_table: dict[frozenset[int], MCGSNode] = {}

    root_region = frozenset({ctx.seed_idx})
    root = MCGSNode(region=root_region)
    transposition_table[root_region] = root

    best_region = root.region
    best_score = -float("inf")

    eval_count = 0
    trajectory: list[dict[str, float]] = []

    for _ in tqdm(range(num_iterations), desc="MCGS", unit="iter"):
        # --- SELECTION + EXPANSION ---
        node = root
        path = [node]
        actions: list[int] = []

        while True:
            if is_terminal(
                node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
            ):
                break

            expansion = expand_node_eager(
                node, ctx.image_graph, ctx.used_segments, transposition_table
            )
            if expansion is not None:
                action, new_child = expansion
                actions.append(action)
                node = new_child
                path.append(node)
                break

            # Fully expanded: descend via UCT
            action, child = select_uct_child(node, exploration_c, alpha)
            actions.append(action)
            node = child
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
        backup_path(path, actions, rewards)

    best_score = best_score * ctx.optimization_sign

    return RegionResult(
        region=best_region,
        score=best_score,
        evaluations_count=eval_count,
        trajectory=trajectory,
    )
