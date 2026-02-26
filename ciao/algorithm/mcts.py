"""Unified Monte Carlo Tree Search for Connected Image Segments.

This module provides two MCTS variants controlled by the 'mode' parameter:
1. 'standard': Standard MCTS with UCT selection and random rollouts
2. 'rave': MCTS with Rapid Action Value Estimation (local + global  RAVE)

Both modes support:
- Batch collection and evaluation for GPU efficiency
- Virtual loss for parallel safety
- Terminal caching to avoid re-evaluating visited states
- MAX backup for finding peak explanations

State = integer bitmask of selected segments
"""

import math
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from ciao.structures.bitmask_graph import (
    add_node,
    get_frontier,
    has_node,
    iter_bits,
    sample_connected_superset,
)
from ciao.structures.nodes import MCTSNode
from ciao.utils.calculations import ModelPredictor
from ciao.utils.search_utils import evaluate_masks, is_terminal


# ============================================================================
# RAVE-specific Classes
# ============================================================================


class GlobalStats:
    """Global RAVE statistics shared across the entire search tree (RAVE mode only)."""

    def __init__(self, num_segments: int):
        self.visits = np.zeros(num_segments, dtype=np.int32)
        self.value_sum = np.zeros(num_segments, dtype=np.float32)

    def get_prior_score(self, seg_id: int) -> float:
        """Get the global RAVE score for a segment (smart FPU)."""
        if self.visits[seg_id] == 0:
            return 0.0
        return self.value_sum[seg_id] / self.visits[seg_id]

    def update(self, rollout_mask: int, reward: float) -> None:
        """Update global stats for all segments in the rollout."""
        for seg_id in iter_bits(rollout_mask):
            self.visits[seg_id] += 1
            self.value_sum[seg_id] += reward


# ============================================================================
# Shared Helper Functions
# ============================================================================


def is_fully_expanded(
    node: MCTSNode, adj_masks: tuple[int, ...], used_mask: int
) -> bool:
    """Check if all frontier segments have been expanded as children."""
    frontier = get_frontier(node.mask, adj_masks, used_mask)

    return all(seg_id in node.children for seg_id in iter_bits(frontier))


# ============================================================================
# Selection Functions
# ============================================================================


def select_uct_child(
    node: MCTSNode, exploration_c: float, virtual_loss: float
) -> MCTSNode | None:
    """Select child with highest UCT score using MAX-value (simple and nested modes)."""
    best_score = -float("inf")
    best_child = None

    parent_visits = (
        node.visits + node.pending * virtual_loss + 1
    )  # +1 for numerical stability

    for child in node.children.values():
        # Virtual loss: increase effective visit count
        effective_visits = child.visits + child.pending * virtual_loss

        # UCT formula with MAX value (not mean)
        exploit = child.max_value if child.visits > 0 else 0.0
        explore = exploration_c * math.sqrt(
            math.log(parent_visits) / max(1, effective_visits)
        )
        score = exploit + explore

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def calculate_beta(visits: int, k: int = 1000) -> float:
    """Calculate beta parameter for MC-RAVE combination (RAVE mode only).

    Beta controls the weight given to RAVE vs. real statistics:
    - β = 1: Trust RAVE completely (at start)
    - β → 0: Trust real statistics more (as visits increase)

    Formula: β = sqrt(k / (3 * visits + k))
    """
    return math.sqrt(k / (3 * visits + k))


def select_uct_child_rave(
    node: MCTSNode, exploration_c: float, virtual_loss: float, rave_k: int
) -> MCTSNode | None:
    """Select child with highest MC-RAVE score (RAVE mode only).

    Combines real statistics (MAX value) with RAVE statistics (mean value)
    using the optimistic beta parameter.
    """
    best_score = -float("inf")
    best_child = None

    parent_visits = (
        node.visits + node.pending * virtual_loss + 1
    )  # +1 for numerical stability

    for child in node.children.values():
        effective_visits = child.visits + child.pending * virtual_loss
        beta = calculate_beta(child.visits, rave_k) if child.visits > 0 else 1.0
        q_real = child.max_value if child.visits > 0 else -float("inf")
        q_rave = (
            (child.rave_value_sum / child.rave_visits) if child.rave_visits > 0 else 0.0
        )

        # Combined Q with Global RAVE for unvisited nodes
        if child.visits == 0:
            # Use Global RAVE prior (smart FPU initialization)
            q_combined = child.prior_score
        else:
            # Use local MC-RAVE: Optimistic combination of real and local RAVE
            q_combined = (1 - beta) * q_real + beta * q_rave

        explore = exploration_c * math.sqrt(
            math.log(parent_visits) / max(1, effective_visits)
        )
        score = q_combined + explore

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


# ============================================================================
# Expansion Functions
# ============================================================================


def expand_node(
    node: MCTSNode,
    adj_masks: tuple[int, ...],
    used_mask: int,
    global_stats: GlobalStats | None = None,
) -> MCTSNode | None:
    """Standard expansion: Pick one random unexpanded segment (standard and RAVE modes).

    Args:
        node: Node to expand
        adj_masks: Adjacency bitmasks
        used_mask: Globally excluded segments
        global_stats: Optional GlobalStats for RAVE prior initialization

    Returns:
        New child node if created, None if no expansion possible
    """
    frontier = get_frontier(node.mask, adj_masks, used_mask)

    unexpanded = []
    for seg_id in iter_bits(frontier):
        if seg_id not in node.children:
            unexpanded.append(seg_id)

    if not unexpanded:
        return None

    # Create one new child
    seg_id = random.choice(unexpanded)
    child_mask = add_node(node.mask, seg_id)

    # Get prior score for RAVE mode
    prior_score = (
        global_stats.get_prior_score(seg_id) if global_stats is not None else 0.0
    )

    child = MCTSNode(mask=child_mask, parent=node, prior_score=prior_score)
    node.children[seg_id] = child

    return child


# ============================================================================
# Backup Functions
# ============================================================================


def backup_paths(batch_paths: list[list[MCTSNode]], rewards: list[float]) -> None:
    """Backup rewards using standard statistics (standard mode).

    Updates:
    - visits, value_sum (mean tracking)
    - max_value (MAX backup for selection)
    - pending (release virtual loss)
    """
    for path, reward in zip(batch_paths, rewards, strict=True):
        for node in path:
            # Release virtual loss
            if node.pending <= 0:
                raise RuntimeError(
                    f"Virtual loss underflow: node.pending={node.pending} (should be > 0)"
                )
            node.pending -= 1
            node.visits += 1
            node.value_sum += reward  # Mean tracking
            node.max_value = max(node.max_value, reward)  # MAX backup


def backup_paths_rave(
    batch_paths: list[list[MCTSNode]],
    batch_rollout_masks: list[int],
    rewards: list[float],
    global_stats: GlobalStats | None = None,
) -> None:
    """Backup rewards using standard, local RAVE, and global RAVE updates (RAVE mode).

    Standard Backup:
        - Update visits and max_value for nodes on the path

    Local RAVE Backup (AMAF):
        - For each node in path, check ALL its children
        - If a child's segment appears in the rollout, update its RAVE stats
        - This generalizes learning: "if we picked X later, it was good/bad"

    Global RAVE Backup:
        - Update global statistics for all segments in the rollout
        - Used for smart FPU initialization of new nodes

    Args:
        batch_paths: List of node paths (one per rollout)
        batch_rollout_masks: List of final rollout masks
        rewards: List of rewards for each rollout
        global_stats: Optional global RAVE statistics
    """
    for path, rollout_mask, reward in zip(
        batch_paths, batch_rollout_masks, rewards, strict=True
    ):
        # --- GLOBAL RAVE BACKUP ---
        if global_stats is not None:
            global_stats.update(rollout_mask, reward)

        for node in path:
            # --- STANDARD BACKUP ---
            # Release virtual loss
            if node.pending <= 0:
                raise RuntimeError(
                    f"Virtual loss underflow: node.pending={node.pending} (should be > 0)"
                )
            node.pending -= 1
            node.visits += 1
            node.value_sum += reward  # Mean tracking
            node.max_value = max(node.max_value, reward)  # MAX backup

            # --- LOCAL RAVE BACKUP (AMAF) ---
            # Update RAVE stats for children that appeared in rollout
            for child_seg_id, child in node.children.items():
                # Check if this child's segment appears in the rollout mask
                if has_node(rollout_mask, child_seg_id):
                    # This segment was used in the rollout
                    child.rave_visits += 1
                    child.rave_value_sum += reward


# ============================================================================
# Main Unified MCTS Function
# ============================================================================


def build_hyperpixel_mcts(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    seed_idx: int,
    desired_length: int,
    num_iterations: int,
    mode: str = "standard",
    optimization_sign: int = 1,
    batch_size: int = 64,
    exploration_c: float = 1.4,
    virtual_loss: float = 1.0,
    used_mask: int = 0,
    # RAVE-specific parameters
    rave_k: int = 1000,
) -> dict[str, Any]:
    """Build a hyperpixel using Monte Carlo Tree Search.

    Supports two modes:
    1. 'standard': Standard MCTS with random rollouts
    2. 'rave': MCTS with Rapid Action Value Estimation (local + global RAVE)

    Both modes include:
    - Batch collection and evaluation
    - Terminal caching to avoid re-evaluation
    - Virtual loss for parallel safety
    - MAX backup for finding peak explanations

    Args:
        predictor: Model for evaluating segment masks
        input_batch: Preprocessed image tensor
        segments: Segmentation map array
        adj_masks: Adjacency bitmasks for each segment
        target_class_idx: Target class to explain
        seed_idx: Starting segment index
        desired_length: Target number of segments per hyperpixel
        num_iterations: Number of MCTS iterations (batch collections)
        mode: MCTS variant - 'standard' or 'rave'
        optimization_sign: +1 to maximize deltas, -1 to minimize
        batch_size: Number of leaf nodes to collect per iteration
        exploration_c: UCT exploration constant
        virtual_loss: Multiplier for pending counter in UCT
        used_mask: Bitmask of globally excluded segments
        rave_k: RAVE equivalence parameter (controls beta decay) - RAVE mode only

    Returns:
        Dict with best mask, score, search statistics, and mode-specific data
    """
    if mode not in ["standard", "rave"]:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'standard' or 'rave'.")

    # Initialize mode-specific components
    global_stats = None
    if mode == "rave":
        num_segments = len(adj_masks)
        global_stats = GlobalStats(num_segments)

    # Create root node
    root_mask = add_node(0, seed_idx)
    root = MCTSNode(mask=root_mask, parent=None)

    best_mask = root.mask
    best_score = -float("inf")
    best_score_history = []

    # Statistics tracking
    total_cache_hits = 0
    total_gpu_evaluations = 0

    # --- MAIN MCTS LOOP ---
    for _ in tqdm(range(num_iterations), desc=f"MCTS-{mode.upper()}", unit="iter"):
        # --- PHASE 1: BATCH COLLECTION ---
        batch_paths = []
        batch_rollout_masks = []
        cached_rewards: list[
            float | None
        ] = []  # Store cached values for visited terminals

        for _ in range(batch_size):
            # --- SELECTION ---
            node = root
            path = [node]

            # Apply virtual loss to root
            root.pending += 1

            # Standard selection for standard and RAVE modes
            while is_fully_expanded(node, adj_masks, used_mask) and not is_terminal(
                node.mask, adj_masks, used_mask, desired_length
            ):
                if mode == "rave":
                    child = select_uct_child_rave(
                        node, exploration_c, virtual_loss, rave_k
                    )
                else:
                    child = select_uct_child(node, exploration_c, virtual_loss)

                if child is None:
                    raise RuntimeError(
                        "Selection failed to find a child, but node is fully expanded."
                    )

                child.pending += 1
                node = child
                path.append(node)

            # --- EXPANSION ---
            if not is_terminal(node.mask, adj_masks, used_mask, desired_length):
                child = expand_node(node, adj_masks, used_mask, global_stats)

                if child is not None:
                    child.pending += 1
                    node = child
                    path.append(node)

            # --- SIMULATION (or cache lookup) ---
            # Check terminal cache
            if (
                is_terminal(node.mask, adj_masks, used_mask, desired_length)
                and node.visits > 0
            ):
                # Reuse cached value - no GPU evaluation needed
                rollout_mask = node.mask
                cached_rewards.append(node.max_value)
            else:
                # Need GPU evaluation
                if is_terminal(node.mask, adj_masks, used_mask, desired_length):
                    rollout_mask = node.mask
                else:
                    # Random rollout
                    frontier = get_frontier(node.mask, adj_masks, used_mask)
                    rollout_mask = sample_connected_superset(
                        base_mask=node.mask,
                        target_length=desired_length,
                        adj_masks=adj_masks,
                        base_frontier=frontier,
                        used_mask=used_mask,
                    )

                cached_rewards.append(None)

            batch_paths.append(path)
            batch_rollout_masks.append(rollout_mask)

        # --- PHASE 2: BATCH EVALUATION ---
        # Separate masks that need GPU evaluation from cached ones
        masks_to_evaluate = [
            (i, batch_rollout_masks[i])
            for i, reward in enumerate(cached_rewards)
            if reward is None
        ]

        # Update statistics
        cache_hits = batch_size - len(masks_to_evaluate)
        total_cache_hits += cache_hits
        total_gpu_evaluations += len(masks_to_evaluate)

        # Evaluate only masks that need GPU
        gpu_rewards = []
        if masks_to_evaluate:
            _indices, masks = zip(*masks_to_evaluate, strict=True)
            raw_rewards = evaluate_masks(
                predictor, input_batch, segments, target_class_idx, list(masks)
            )
            gpu_rewards = [r * optimization_sign for r in raw_rewards]

        # Merge GPU results with cached values (cached values are already signed)
        batch_rewards: list[float] = []
        gpu_idx = 0

        for cached_val in cached_rewards:
            if cached_val is not None:
                batch_rewards.append(cached_val)
            else:
                batch_rewards.append(gpu_rewards[gpu_idx])
                gpu_idx += 1

        # Update best mask if we found a better one
        for i, reward in enumerate(batch_rewards):
            if reward > best_score:
                best_score = reward
                best_mask = batch_rollout_masks[i]

        # --- PHASE 3: BATCH BACKUP ---
        if mode == "rave":
            backup_paths_rave(
                batch_paths=batch_paths,
                batch_rollout_masks=batch_rollout_masks,
                rewards=batch_rewards,
                global_stats=global_stats,
            )
        else:
            backup_paths(batch_paths, batch_rewards)

        # Track best score after each iteration
        best_score_history.append(best_score * optimization_sign)

    best_score = best_score * optimization_sign

    # Count total nodes created
    def count_nodes(node: MCTSNode) -> int:
        return 1 + sum(count_nodes(c) for c in node.children.values())

    # Update used_mask with segments from best mask
    updated_used_mask = used_mask
    for seg_id in iter_bits(best_mask):
        updated_used_mask = add_node(updated_used_mask, seg_id)

    # Build return dictionary with mode-specific stats
    result = {
        "mask": best_mask,
        "score": best_score,
        "used_mask": updated_used_mask,
        "stats": {
            "method": "mcts",
            "mode": mode,
            "iterations": num_iterations,
            "batch_size": batch_size,
            "total_evaluations": num_iterations * batch_size,
            "gpu_evaluations": total_gpu_evaluations,
            "cache_hits": total_cache_hits,
            "cache_hit_rate": total_cache_hits / (num_iterations * batch_size)
            if num_iterations * batch_size > 0
            else 0,
            "best_score_history": best_score_history,
            "nodes": count_nodes(root),
            "root_visits": root.visits,
        },
        "root": root,
    }

    # Add RAVE-specific data
    if mode == "rave":
        result["stats"]["rave_k"] = rave_k  # type: ignore[index]

    return result


def build_all_hyperpixels_mcts(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    scores: dict[int, float],
    max_hyperpixels: int = 10,
    desired_length: int = 30,
    num_iterations: int = 100,
    mode: str = "standard",
    batch_size: int = 64,
    exploration_c: float = 1.4,
    virtual_loss: float = 1.0,
    rave_k: int = 1000,
) -> list[dict[str, object]]:
    """Build multiple hyperpixels using MCTS.

    Args:
        predictor: Model predictor
        input_batch: Preprocessed input tensor
        segments: Segmentation map
        adj_masks: Adjacency bitmasks
        target_class_idx: Target class index
        scores: Individual segment scores
        max_hyperpixels: Maximum number of hyperpixels to build
        desired_length: Target segments per hyperpixel
        num_iterations: Number of MCTS iterations
        mode: 'standard' or 'rave'
        batch_size: Batch size for evaluation
        exploration_c: UCT exploration constant
        virtual_loss: Virtual loss multiplier for parallel safety
        rave_k: RAVE parameter

    Returns:
        List of hyperpixel dictionaries
    """
    hyperpixels = []
    processed_segments = set()
    used_mask = 0

    for _ in range(max_hyperpixels):
        available_segments = [
            seg_id for seg_id in scores if seg_id not in processed_segments
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        result = build_hyperpixel_mcts(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            adj_masks=adj_masks,
            target_class_idx=target_class_idx,
            seed_idx=seed_idx,
            desired_length=desired_length,
            num_iterations=num_iterations,
            mode=mode,
            optimization_sign=optimization_sign,
            batch_size=batch_size,
            exploration_c=exploration_c,
            virtual_loss=virtual_loss,
            used_mask=used_mask,
            rave_k=rave_k,
        )

        hyperpixel_mask = result["mask"]
        used_mask = result["used_mask"]
        hyperpixel_segments = list(iter_bits(hyperpixel_mask))

        if hyperpixel_segments:
            hyperpixels.append(
                {
                    "segments": hyperpixel_segments,
                    "sign": optimization_sign,
                    "size": len(hyperpixel_segments),
                    "hyperpixel_score": result["score"],
                    "stats": result["stats"],  # Include MCTS search statistics
                }
            )
            processed_segments.update(hyperpixel_segments)

    return hyperpixels
