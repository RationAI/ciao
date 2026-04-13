"""Monte Carlo Tree Search for Connected Image Segments.

Supports:
- Batch collection and evaluation for GPU efficiency
- Virtual loss for parallel safety
- Terminal caching to avoid re-evaluating visited states
- MAX backup for finding peak explanations

State = integer bitmask of selected segments
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

    return all(seg_id in node.children for seg_id in frontier)


def select_uct_child(
    node: MCTSNode,
    exploration_c: float,
    virtual_loss: float,
    best_score: float,
    worst_score: float,
) -> MCTSNode | None:
    """Select child with highest UCT score using Min-Max Normalized Q-value and Virtual Loss."""
    best_uct = -float("inf")
    best_child = None

    parent_visits = node.visits + node.pending

    for child in node.children.values():
        # Min-max normalize max_value to [0, 1]
        if (
            child.visits > 0
            and best_score != -float("inf")
            and worst_score != float("inf")
        ):
            q_bar = (child.max_value - worst_score) / (best_score - worst_score + 1e-6)
            q_bar = max(0.0, min(1.0, q_bar))  # clip to [0, 1]
        else:
            q_bar = 1.0  # Optimistic initialization for unvisited nodes

        # Exploration term
        explore = exploration_c * math.sqrt(
            math.log(max(1, parent_visits)) / max(1, child.visits + child.pending)
        )

        # UCT formula combination - TODO: is subtracting that term necessary?
        score = max(0, q_bar - (child.pending * virtual_loss)) + explore

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

    unexpanded = []
    for seg_id in frontier:
        if seg_id not in node.children:
            unexpanded.append(seg_id)

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
    - max_value (MAX backup for selection)
    - pending (release virtual loss)
    """
    for path, reward in zip(batch_paths, rewards, strict=True):
        for node in path[1:]:  # Skip root (never incremented, so shouldn't decrement)
            node.pending -= 1  # Release virtual loss
            node.visits += 1
            node.max_value = max(node.max_value, reward)  # MAX backup


def _collect_mcts_batch(
    ctx: SearchContext,
    root: MCTSNode,
    exploration_c: float,
    virtual_loss: float,
    best_score: float,
    worst_score: float,
) -> tuple[list[list[MCTSNode]], list[frozenset[int]], list[float | None]]:
    """Phase 1: Batch Collection - Selection, Expansion, and Simulation."""
    batch_paths: list[list[MCTSNode]] = []
    batch_rollout_regions: list[frozenset[int]] = []
    cached_rewards: list[float | None] = []  # Store cached values for visited terminals

    for _ in range(ctx.batch_size):
        # --- SELECTION ---
        node = root
        path = [node]

        # Standard selection using MAX UCT
        while is_fully_expanded(
            node, ctx.image_graph, ctx.used_segments
        ) and not is_terminal(
            node.region, ctx.image_graph, ctx.used_segments, ctx.desired_length
        ):
            child = select_uct_child(
                node, exploration_c, virtual_loss, best_score, worst_score
            )

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
            cached_rewards.append(node.max_value)
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
) -> list[float]:
    """Phase 2: Batch Evaluation - Evaluate regions needing GPU and merge carefully."""
    regions_to_evaluate = [
        (i, batch_rollout_regions[i])
        for i, reward in enumerate(cached_rewards)
        if reward is None
    ]

    # Evaluate only regions that need GPU
    gpu_rewards = []
    if regions_to_evaluate:
        _indices, regions = zip(*regions_to_evaluate, strict=True)
        raw_rewards = calculate_region_deltas(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            segments=ctx.image_graph.segments,
            replacement_image=ctx.replacement_image,
            segment_sets=list(regions),
            target_class_idx=ctx.target_class_idx,
            batch_size=ctx.batch_size,
        )
        gpu_rewards = [r * ctx.optimization_sign for r in raw_rewards]

    # Merge GPU results with cached values (cached values are already signed)
    batch_rewards: list[float] = []
    gpu_idx = 0

    for cached_val in cached_rewards:
        if cached_val is not None:
            batch_rewards.append(cached_val)
        else:
            batch_rewards.append(gpu_rewards[gpu_idx])
            gpu_idx += 1

    return batch_rewards


def build_region_mcts(
    ctx: SearchContext,
    num_iterations: int,
    exploration_c: float = 1.4,
    virtual_loss: float = 1.0,
) -> RegionResult:
    """Build a region using Monte Carlo Tree Search.

    Includes:
    - Batch collection and evaluation
    - Terminal caching to avoid re-evaluation
    - Virtual loss for parallel safety
    - MAX backup for finding peak explanations

    Args:
        ctx: Search context with model state and parameters
        num_iterations: Number of MCTS iterations (batch collections)
        exploration_c: UCT exploration constant
        virtual_loss: Multiplier for pending counter in UCT

    Returns:
        RegionResult with region, score, and stats.
    """
    # Create root node
    root_region = frozenset({ctx.seed_idx})
    root = MCTSNode(region=root_region, parent=None)

    best_region = root.region
    best_score = -float("inf")
    worst_score = float("inf")

    # --- MAIN MCTS LOOP ---
    for _ in tqdm(range(num_iterations), desc="MCTS", unit="iter"):
        # --- PHASE 1: BATCH COLLECTION ---
        batch_paths, batch_rollout_regions, cached_rewards = _collect_mcts_batch(
            ctx=ctx,
            root=root,
            exploration_c=exploration_c,
            virtual_loss=virtual_loss,
            best_score=best_score,
            worst_score=worst_score,
        )

        # --- PHASE 2: BATCH EVALUATION ---
        batch_rewards = _evaluate_mcts_batch(
            ctx=ctx,
            batch_rollout_regions=batch_rollout_regions,
            cached_rewards=cached_rewards,
        )

        # Update best region if we found a better one
        for i, reward in enumerate(batch_rewards):
            if reward > best_score:
                best_score = reward
                best_region = batch_rollout_regions[i]
            if reward < worst_score:
                worst_score = reward

        # --- PHASE 3: BATCH BACKUP ---
        backup_paths(batch_paths, batch_rewards)

    best_score = best_score * ctx.optimization_sign

    return RegionResult(region=best_region, score=best_score)
