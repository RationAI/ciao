"""Bitmask-based graph utilities for efficient segment manipulation.

This module will provide low-level primitives for working with graph structures
represented as integer bitmasks, where each bit represents a node/segment.
"""

from collections.abc import Iterator


def iter_bits(mask: int) -> Iterator[int]:
    """Iterate over set bits in a mask using low-bit isolation.

    Yields node IDs in arbitrary order (depends on bit positions).
    Performance: O(k) where k is the number of set bits.

    Example:
        mask = 0b10110  # bits 1, 2, 4 are set
        list(iter_bits(mask))  # [1, 2, 4]
    """
    if mask < 0:
        raise ValueError("mask cannot be negative")

    temp = mask
    while temp:
        low_bit = temp & -temp
        node_id = low_bit.bit_length() - 1
        yield node_id
        temp ^= low_bit


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
    if mask < 0 or used_mask < 0:
        raise ValueError("mask and used_mask cannot be negative")

    frontier = 0

    for node_id in iter_bits(mask):
        frontier |= adj_masks[node_id]

    frontier &= ~mask
    frontier &= ~used_mask

    return frontier
