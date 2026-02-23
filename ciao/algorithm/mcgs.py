"""Unified Monte Carlo Graph Search (MCGS) Implementation

This module provides a unified MCGS implementation with multiple modes:
- mode='standard': Standard MCGS with eager expansion and optimized node creation
- mode='rave': RAVE (Rapid Action Value Estimation) with edge-level statistics

All modes share:
- Tree structure for state exploration
- Eager expansion strategy (check entire frontier before creating nodes)
- Virtual loss for parallel batch safety
- MAX-based UCT for deterministic optimization

Key features:
- MCGSNode: Tree nodes with edge-level statistics (for RAVE mode)
- use_guided_rollout: Choose between guided (weighted) vs pure random rollouts
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
    iter_bits,
    sample_connected_superset,
)
from ciao.utils.calculations import ModelPredictor
from ciao.structures.nodes import MCGSNode
from ciao.utils.search_utils import evaluate_masks, is_terminal


def select_uct_child(
    node: MCGSNode, exploration_c: float, virtual_loss: float
) -> tuple[int, MCGSNode] | None:
    """Select child with highest UCT score using edge statistics (MCGS mode).
    
    In MCGS, we use edge statistics rather than node statistics to handle DAGs correctly.
    A child node may have high visits from other parents, which should not influence
    selection from this parent.
    
    Returns:
        (action, child) tuple with best UCT score, or None if no children
    """
    if not node.children:
        return None

    best_score = -float("inf")
    best_action = None
    best_child = None

    parent_visits = node.visits + 1  # +1 for numerical stability

    for action, child in node.children.items():
        # Get edge statistics (with virtual loss)
        edge_stats = node.edge_stats.get(
            action, {"N": 0, "W": 0.0, "Q": 0.0, "max_reward": -float("inf")}
        )
        pending = node.pending_edges.get(action, 0)
        
        # Use edge visit count (not child.visits) with virtual loss
        edge_n = edge_stats["N"] + pending * virtual_loss

        # Exploitation: Use edge max_reward (not child.max_value)
        exploit = edge_stats["max_reward"] if edge_stats["N"] > 0 else 0.0
        
        # Exploration: Use edge visit count in denominator
        explore = exploration_c * math.sqrt(
            math.log(parent_visits) / max(1, edge_n)
        )
        
        score = exploit + explore

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is not None and best_action is not None:
        return (best_action, best_child)
    return None


def select_uct_child_rave(
    node: MCGSNode, exploration_c: float, virtual_loss: float, rave_k: float
) -> tuple[int, MCGSNode] | None:
    """Select child with highest UCT score using RAVE mixing.

    RAVE formula:
        beta = sqrt(k / (3 * N_edge + k))
        Q_combined = (1 - beta) * Q_edge + beta * Q_rave
        UCT = Q_combined + c * sqrt(log(N_parent) / N_edge)

    Returns:
        (action, child) tuple with best UCT score, or None if no children
    """
    if not node.children:
        return None

    best_score = -float("inf")
    best_action = None
    best_child = None

    parent_visits = node.visits + 1  # +1 for numerical stability

    for action, child in node.children.items():
        # Get edge statistics (with virtual loss)
        edge_stats = node.edge_stats.get(
            action, {"N": 0, "W": 0.0, "Q": 0.0, "max_reward": -float("inf")}
        )
        rave_stats = node.rave_stats.get(action, {"N": 0, "W": 0.0, "Q": 0.0})
        pending = node.pending_edges.get(action, 0)
        edge_n = edge_stats["N"] + pending * virtual_loss

        # RAVE mixing
        # Use actual edge statistics only if we have real visits (not just virtual loss)
        if edge_stats["N"] > 0:
            # Calculate beta (mixing parameter)
            beta = math.sqrt(rave_k / (3 * edge_n + rave_k))

            # Get Q values:
            # - For MC (edge): Use MAX reward (deterministic optimization)
            # - For RAVE: Use MEAN reward (AMAF is a heuristic average)
            q_edge = edge_stats["max_reward"]
            q_rave = rave_stats["Q"] if rave_stats["N"] > 0 else 0.0

            # Combined Q value
            q_combined = (1 - beta) * q_edge + beta * q_rave
        else:
            # No edge visits yet, use pure RAVE or 0
            q_combined = rave_stats["Q"] if rave_stats["N"] > 0 else 0.0

        # UCT exploration term (uses edge visits, not child.visits)
        explore = exploration_c * math.sqrt(math.log(parent_visits) / max(1, edge_n))

        # Final UCT score
        score = q_combined + explore

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is not None and best_action is not None:
        return (best_action, best_child)
    return None


def expand_node_eager(
    node: MCGSNode,
    adj_masks: tuple,
    used_mask: int,
    transposition_table: dict[int, MCGSNode],
    mode: str,
) -> tuple[int, MCGSNode] | None:
    """Eager expansion with grafting: Check ALL frontier segments before creating a new node.

    This function processes the entire frontier to:
    1. Skip segments that are already children
    2. Link existing nodes from transposition table (grafting)
    3. Identify truly new candidates for node creation
    4. Randomly select one new candidate to expand

    Returns:
        (segment_id, child_node) if a NEW node was created
        None if all frontier segments are already children or grafted
    """
    frontier = get_frontier(node.mask, adj_masks, used_mask)

    if frontier == 0:
        return None

    existing_children_ids = set(node.children.keys())

    new_candidates = []

    for seg_id in iter_bits(frontier):
        # Check: Already a child?
        if seg_id in existing_children_ids:
            continue  # Already linked, skip

        new_mask = add_node(node.mask, seg_id)

        # Check: Does this state already exist in the graph?
        if new_mask in transposition_table:
            # GRAFT: Link existing node (DAG structure)
            existing_node = transposition_table[new_mask]
            node.children[seg_id] = existing_node
            if mode == "rave":
                node.init_edge(seg_id)
        else:
            # Truly new: add to candidates for potential creation
            new_candidates.append((seg_id, new_mask))

    if not new_candidates:
        # All frontier segments are already children or grafted
        return None

    # Pick one random new candidate and create it
    seg_id, new_mask = random.choice(new_candidates)
    child = MCGSNode(mask=new_mask)
    transposition_table[new_mask] = child
    node.children[seg_id] = child
    if mode == "rave":
        node.init_edge(seg_id)

    return seg_id, child


def update_edge_stats(node: MCGSNode, action: int, reward: float):
    """Update edge statistics for a specific action (RAVE mode)."""
    if action not in node.edge_stats:
        node.init_edge(action)

    stats = node.edge_stats[action]
    stats["N"] += 1
    stats["W"] += reward
    stats["Q"] = stats["W"] / stats["N"]
    stats["max_reward"] = max(stats["max_reward"], reward)


def update_rave_stats(node: MCGSNode, action: int, reward: float):
    """Update RAVE statistics for a specific action (RAVE mode)."""
    if action not in node.rave_stats:
        node.init_edge(action)

    stats = node.rave_stats[action]
    stats["N"] += 1
    stats["W"] += reward
    stats["Q"] = stats["W"] / stats["N"]
    stats["max_reward"] = max(stats.get("max_reward", -float("inf")), reward)


def backup_paths(
    batch_paths: list[list[MCGSNode]], 
    batch_actions: list[list[int]],
    rewards: list[float]
) -> None:
    """Backup rewards through all nodes in the paths (standard mode).

    Updates:
    - visits, value_sum (mean tracking)
    - max_value (MAX backup for selection)
    - edge statistics (track edge-level stats for proper DAG handling)
    - pending_edges (release virtual loss)

    Args:
        batch_paths: List of node paths (one per simulation)
        batch_actions: List of action sequences (one per simulation)
        rewards: List of rewards for each path
    """
    for path, actions, reward in zip(batch_paths, batch_actions, rewards):
        for i, node in enumerate(path):
            # Update node statistics
            node.visits += 1
            node.value_sum += reward  # Mean tracking
            node.max_value = max(node.max_value, reward)  # MAX backup
            
            # Update edge statistics and release virtual loss
            if i > 0:  # Skip root (no incoming edge)
                action = actions[i - 1]  # Action that led to this node
                parent = path[i - 1]
                
                # Release virtual loss on edge
                if action in parent.pending_edges:
                    parent.pending_edges[action] = max(0, parent.pending_edges[action] - 1)
                
                # Update edge statistics
                update_edge_stats(parent, action, reward)


def backup_paths_rave(
    batch_paths: list[list[MCGSNode]],
    batch_actions: list[list[int]],
    batch_masks: list[int],
    rewards: list[float],
    adj_masks: tuple,
    used_mask: int,
) -> None:
    """Backup rewards using edge-level statistics and RAVE updates (RAVE mode).

    Standard Backup:
        - Update visits and max_value for nodes on the path

    Edge-level Backup:
        - Update edge statistics for the action taken from each node

    RAVE Backup (AMAF - All-Moves-As-First):
        - For each node in path, check ALL its frontier segments
        - If a segment appears in the rollout, update its RAVE stats
        - This generalizes learning: "if we picked X later, it was good/bad"

    Args:
        batch_paths: List of node paths (one per simulation)
        batch_actions: List of action sequences (one per path)
        batch_masks: List of final rollout masks
        rewards: List of rewards for each path
        adj_masks: Adjacency bitmasks for frontier calculation
        used_mask: Globally excluded segments
    """
    for path, actions, rollout_mask, reward in zip(
        batch_paths, batch_actions, batch_masks, rewards
    ):
        rollout_segments = set(iter_bits(rollout_mask))

        for i, node in enumerate(path):
            # --- STANDARD BACKUP ---
            node.visits += 1
            node.value_sum += reward
            node.max_value = max(node.max_value, reward)

            # --- EDGE-LEVEL BACKUP ---
            # Update edge statistics for the action taken from this node
            if i < len(actions):
                action = actions[i]
                update_edge_stats(node, action, reward)

            # --- RAVE BACKUP (AMAF) ---
            # Update RAVE stats for frontier segments that appeared in rollout
            # Optimized: iterate through rollout_segments (small) instead of frontier (large)
            frontier = get_frontier(node.mask, adj_masks, used_mask)
            frontier_bits = frontier  # Keep as bitmask for fast membership check
            for seg_id in rollout_segments:
                # Check if this rollout segment was legal from this node
                if (frontier_bits >> seg_id) & 1:  # Fast bit check
                    update_rave_stats(node, seg_id, reward)

            # --- RELEASE VIRTUAL LOSS ---
            if i < len(actions):
                action = actions[i]
                if action in node.pending_edges:
                    node.pending_edges[action] = max(0, node.pending_edges[action] - 1)


def build_hyperpixel_mcgs(
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
    rave_k: float = 1000.0,
) -> dict[str, Any]:
    """Unified Monte Carlo Graph Search for hyperpixel selection.

    This function implements two MCGS modes:
    - 'standard': Standard MCGS with eager expansion and optimized grafting
    - 'rave': RAVE with edge-level statistics

    Args:
        predictor: Model for evaluating segment masks
        input_batch: Preprocessed image tensor
        segments: Segmentation map array
        adj_masks: Adjacency bitmasks for each segment
        target_class_idx: Target class to explain
        seed_idx: Starting segment index
        desired_length: Maximum hyperpixel size
        num_iterations: Number of MCGS iterations
        mode: 'standard' or 'rave'
        optimization_sign: +1 to maximize deltas, -1 to minimize
        batch_size: Number of paths to collect per iteration
        exploration_c: UCT exploration constant
        virtual_loss: Multiplier for pending counter in UCT
        used_mask: Bitmask of globally excluded segments
        rave_k: RAVE mixing parameter (for mode="rave")

    Returns:
        Dict with best mask, score, and search statistics
    """
    if mode not in ["standard", "rave"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'standard' or 'rave'")

    # --- INITIALIZATION ---
    # Create transposition table for state deduplication (DAG structure)
    transposition_table = {}

    # Create root node
    root_mask = add_node(0, seed_idx)
    root = MCGSNode(mask=root_mask)
    transposition_table[root_mask] = root

    best_mask = root.mask
    best_score = -float("inf")
    best_score_history = []

    # Statistics tracking
    total_cache_hits = 0
    total_gpu_evaluations = 0

    # --- MAIN MCGS LOOP ---
    mode_label = f"MCGS-{mode.upper()}"
    for iteration in tqdm(range(num_iterations), desc=f"  {mode_label}", ncols=80):
        # --- PHASE 1: BATCH COLLECTION ---
        batch_paths = []
        batch_masks = []
        cached_rewards = []  # Store cached values for visited terminals
        needs_gpu_eval = []  # Track which entries need GPU evaluation
        batch_actions = []  # For RAVE mode: track actions taken

        for _ in range(batch_size):
            # --- SELECTION with EAGER EXPANSION ---
            node = root
            path = [node]
            actions_taken = []  # Track actions for RAVE

            # Continue descending until we create a new node or reach terminal
            while (
                expansion_result := expand_node_eager(
                    node, adj_masks, used_mask, transposition_table, mode
                )
            ) is None and not is_terminal(
                node.mask, adj_masks, used_mask, desired_length
            ):
                # All frontier segments are already children or grafted - select best child
                if mode == "rave":
                    uct_result = select_uct_child_rave(
                        node, exploration_c, virtual_loss, rave_k
                    )
                    assert uct_result is not None
                    action, child = uct_result

                    # Apply virtual loss to edge
                    if action not in node.pending_edges:
                        node.pending_edges[action] = 0
                    node.pending_edges[action] += 1

                    actions_taken.append(action)
                else:
                    uct_result = select_uct_child(node, exploration_c, virtual_loss)
                    assert uct_result is not None
                    action, child = uct_result

                    # Apply virtual loss to edge
                    if action not in node.pending_edges:
                        node.pending_edges[action] = 0
                    node.pending_edges[action] += 1
                    
                    actions_taken.append(action)

                node = child
                path.append(node)

            # --- EXPANSION ---
            if expansion_result is not None:
                seg_id, child = expansion_result

                # Apply virtual loss to edge (both modes)
                if seg_id not in node.pending_edges:
                    node.pending_edges[seg_id] = 0
                node.pending_edges[seg_id] += 1
                actions_taken.append(seg_id)

                node = child
                path.append(node)

            # --- SIMULATION: Generate rollout mask ---
            if (
                is_terminal(node.mask, adj_masks, used_mask, desired_length)
                and node.visits > 0
            ):
                rollout_mask = node.mask
                cached_rewards.append(node.max_value)
                needs_gpu_eval.append(False)
            else:
                if is_terminal(node.mask, adj_masks, used_mask, desired_length):
                    rollout_mask = node.mask
                else:
                    # Random rollout using sample_connected_superset
                    frontier = get_frontier(node.mask, adj_masks, used_mask)
                    rollout_mask = sample_connected_superset(
                        base_mask=node.mask,
                        target_length=desired_length,
                        adj_masks=adj_masks,
                        base_frontier=frontier,
                        used_mask=used_mask,
                    )

                cached_rewards.append(None)
                needs_gpu_eval.append(True)

            batch_paths.append(path)
            batch_masks.append(rollout_mask)
            batch_actions.append(actions_taken)

        # --- PHASE 2: BATCH EVALUATION ---
        # Separate masks that need GPU evaluation from cached ones
        masks_to_evaluate = [
            (i, batch_masks[i])
            for i, need_eval in enumerate(needs_gpu_eval)
            if need_eval
        ]

        # Update statistics
        cache_hits = sum(1 for need_eval in needs_gpu_eval if not need_eval)
        total_cache_hits += cache_hits
        total_gpu_evaluations += len(masks_to_evaluate)

        # Evaluate only masks that need GPU
        gpu_rewards = []
        if masks_to_evaluate:
            indices, masks = zip(*masks_to_evaluate)
            raw_rewards = evaluate_masks(
                predictor, input_batch, segments, target_class_idx, list(masks)
            )
            gpu_rewards = [r * optimization_sign for r in raw_rewards]

        # Merge GPU results with cached values (cached values are already signed)
        batch_rewards = []
        gpu_idx = 0
        for i in range(batch_size):
            if not needs_gpu_eval[i]:
                batch_rewards.append(cached_rewards[i])
            else:
                # Use GPU result
                batch_rewards.append(gpu_rewards[gpu_idx])
                gpu_idx += 1

        # Update best score
        for path_idx, (reward, rollout_mask) in enumerate(
            zip(batch_rewards, batch_masks)
        ):
            if reward > best_score:
                best_score = reward
                best_mask = rollout_mask

        # --- PHASE 3: BACKPROPAGATION ---
        if mode == "rave":
            backup_paths_rave(
                batch_paths=batch_paths,
                batch_actions=batch_actions,
                batch_masks=batch_masks,
                rewards=batch_rewards,
                adj_masks=adj_masks,
                used_mask=used_mask,
            )
        else:
            backup_paths(batch_paths, batch_actions, batch_rewards)

        # Track best score history
        best_score_history.append(best_score * optimization_sign)

    # --- RETURN RESULTS ---
    # Convert best score back to raw (un-signed) value
    best_score_raw = best_score * optimization_sign

    # Update used_mask with the segments from the best hyperpixel
    updated_used_mask = used_mask
    for seg_id in iter_bits(best_mask):
        updated_used_mask |= 1 << seg_id

    result = {
        "mask": best_mask,
        "score": best_score_raw,
        "used_mask": updated_used_mask,
        "root": root,
        "stats": {
            "method": "mcgs",
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
            "nodes": len(transposition_table),
            "root_visits": root.visits,
        },
    }

    # Add RAVE-specific data
    if mode == "rave":
        result["stats"]["rave_k"] = rave_k

    return result


def build_all_hyperpixels_mcgs(
    predictor,
    input_batch,
    segments,
    adj_masks,
    target_class_idx,
    scores,
    next_id,
    max_hyperpixels=10,
    desired_length=30,
    num_iterations=100,
    mode="standard",
    batch_size=64,
    exploration_c=1.4,
    virtual_loss=1.0,
    rave_k=1000.0,
):
    """Build multiple hyperpixels using MCGS.

    Args:
        predictor: Model predictor
        input_batch: Preprocessed input tensor
        segments: Segmentation map
        adj_masks: Adjacency bitmasks
        target_class_idx: Target class index
        scores: Individual segment scores
        next_id: Total number of segments
        max_hyperpixels: Maximum number of hyperpixels to build
        desired_length: Target segments per hyperpixel
        num_iterations: Number of MCGS iterations
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

    for i in range(max_hyperpixels):
        available_segments = [
            seg_id for seg_id in scores.keys() if seg_id not in processed_segments
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        result = build_hyperpixel_mcgs(
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
        hyperpixel_segments = [
            seg_id for seg_id in range(next_id) if hyperpixel_mask & (1 << seg_id)
        ]

        if hyperpixel_segments:
            hyperpixels.append(
                {
                    "segments": hyperpixel_segments,
                    "sign": optimization_sign,
                    "size": len(hyperpixel_segments),
                    "hyperpixel_score": result["score"],
                    "stats": result["stats"],  # Include MCGS search statistics
                }
            )
            processed_segments.update(hyperpixel_segments)

    return hyperpixels
