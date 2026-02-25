"""Bitmask-based graph utilities for efficient segment manipulation.

This module provides low-level primitives for working with graph structures
represented as integer bitmasks, where each bit represents a node/segment.
"""

import random
from collections.abc import Iterator

import numpy as np


def mask_to_ids(mask: int) -> list[int]:
    """Convert integer bitmask to list of segment indices."""
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


def iter_bits(mask: int) -> Iterator[int]:
    """Iterate over set bits in a mask using low-bit isolation.

    Yields node IDs in arbitrary order (depends on bit positions).
    Performance: O(k) where k is the number of set bits.

    Example:
        mask = 0b10110  # bits 1, 2, 4 are set
        list(iter_bits(mask))  # [1, 2, 4]
    """
    temp = mask
    while temp:
        low_bit = temp & -temp
        node_id = low_bit.bit_length() - 1
        yield node_id
        temp ^= low_bit


def has_node(mask: int, node: int) -> bool:
    """Test if a node is present in the mask."""
    return (mask >> node) & 1 == 1


def add_node(mask: int, node: int) -> int:
    """Add a node to the mask."""
    return mask | (1 << node)


def remove_node(mask: int, node: int) -> int:
    """Remove a node from the mask."""
    return mask & ~(1 << node)


def pick_random_set_bit(mask: int) -> int:
    """Select a random set bit from the mask in O(N) where N is the index of the bit.

    Without allocating a list. Efficient for sparse masks.
    """
    count = mask.bit_count()
    if count == 0:
        return -1

    which = random.randrange(count)

    temp = mask
    for _ in range(which):
        temp &= temp - 1  # Clear lowest set bit

    return (temp & -temp).bit_length() - 1


def get_frontier(mask: int, adj_masks: tuple[int, ...], used_mask: int) -> int:
    """Compute the expansion frontier (valid neighbors) for graph traversal.

    The frontier is the set of segments adjacent to the current structure
    that can be added in the next step.

    A segment is in the frontier if:
    - It is adjacent to at least one segment in the current mask
    - It is NOT already in the current mask
    - It is NOT in the used_mask (respects global exclusion constraints)

    Args:
        mask: Bitmask of current structure/connected component
        adj_masks: Tuple of adjacency bitmasks (adj_masks[i] = neighbors of segment i)
        used_mask: Bitmask of globally excluded segments

    Returns:
        Bitmask of valid frontier segments
    """
    frontier = 0

    for node_id in iter_bits(mask):
        frontier |= adj_masks[node_id]

    frontier &= ~mask
    frontier &= ~used_mask

    return frontier


def sample_connected_superset(
    base_mask: int,
    target_length: int,
    adj_masks: tuple[int, ...],
    base_frontier: int,
    used_mask: int,
    segment_weights: np.ndarray | None = None,
    optimization_sign: int = 1,
    temperature: float = 3.0,
) -> int:
    """Sample a connected superset via random walk expansion.

    IMPORTANT: This is NOT a uniform sampler over all connected supersets.
    The distribution is biased towards segments discovered early and
    depends on graph topology. This bias is acceptable for Monte Carlo
    estimation in the parent algorithm.

    With segment_weights provided, uses guided sampling based on max rewards
    observed during search, using softmax with temperature and epsilon-mixing
    for exploration.

    Args:
        base_mask: Starting set (must be non-empty and connected)
        target_length: Desired size of the superset
        adj_masks: Adjacency bitmasks for neighbor lookups
        base_frontier: Initial expansion frontier (unused, kept for compatibility)
        used_mask: Global exclusion mask (segments that must not be added)
        segment_weights: Optional array of max rewards per segment for guided sampling
        optimization_sign: +1 to maximize, -1 to minimize (affects weighting)
        temperature: Softmax temperature (higher = more uniform, default 3.0)

    Returns:
        Bitmask of connected superset containing base_mask
    """
    mask = base_mask

    while mask.bit_count() < target_length:
        frontier = get_frontier(mask, adj_masks, used_mask)
        if frontier == 0:
            break

        # Select next segment (weighted or uniform)
        if segment_weights is not None:
            seg_id = _pick_weighted_frontier_segment(
                frontier, segment_weights, optimization_sign, temperature
            )
        else:
            seg_id = pick_random_set_bit(frontier)

        mask = add_node(mask, seg_id)

    return mask


def _pick_weighted_frontier_segment(
    frontier: int,
    segment_weights: np.ndarray,
    optimization_sign: int,
    temperature: float,
) -> int:
    """Pick a segment from frontier using softmax weighting over max rewards.

    Logic:
    1. Extract weights for frontier segments
    2. Replace -inf (unvisited) with min observed reward
    3. Apply optimization sign and softmax with temperature
    4. Mix with uniform distribution (epsilon=0.05) for exploration
    5. Sample using the final probabilities

    Args:
        frontier: Bitmask of candidate segments
        segment_weights: Array of max rewards per segment (may contain -inf)
        optimization_sign: +1 to maximize, -1 to minimize
        temperature: Softmax temperature for probability distribution

    Returns:
        Selected segment ID
    """
    # Extract frontier segment IDs and their weights
    frontier_ids = list(iter_bits(frontier))
    frontier_weights = segment_weights[frontier_ids]

    # Handle unvisited segments: replace -inf with min observed reward
    visited_mask = np.isfinite(frontier_weights)
    if np.any(visited_mask):
        min_observed = np.min(frontier_weights[visited_mask])
        frontier_weights = np.where(visited_mask, frontier_weights, min_observed)
    else:
        # No segments visited yet - treat all as equal (zero)
        frontier_weights = np.zeros_like(frontier_weights)

    # Apply optimization sign to align with "bigger is better"
    effective_scores = frontier_weights * optimization_sign

    # Compute softmax probabilities with temperature
    # Subtract max for numerical stability
    max_score = np.max(effective_scores)
    exp_scores = np.exp((effective_scores - max_score) / temperature)
    softmax_probs = exp_scores / np.sum(exp_scores)

    # Epsilon-greedy mixing: 95% softmax, 5% uniform
    epsilon = 0.05
    uniform_probs = np.ones(len(frontier_ids)) / len(frontier_ids)
    final_probs = (1 - epsilon) * softmax_probs + epsilon * uniform_probs

    # Renormalize to ensure probabilities sum to exactly 1.0 (fix floating point errors)
    final_probs = final_probs / np.sum(final_probs)

    # Sample segment using final probabilities
    idx = np.random.choice(len(frontier_ids), p=final_probs)
    return frontier_ids[idx]
