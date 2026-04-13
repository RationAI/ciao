"""Shared utilities for MCTS and MCGS search algorithms.

This module contains common functions used by both Monte Carlo Tree Search (MCTS)
and Monte Carlo Graph Search (MCGS) implementations.
"""

from collections.abc import Set

from ciao.algorithm.graph import ImageGraph


def is_terminal(
    current_region: Set[int],
    image_graph: ImageGraph,
    used_segments: Set[int],
    max_depth: int,
) -> bool:
    """Check if state is terminal (max depth or no frontier)."""
    return len(current_region) >= max_depth or not image_graph.get_frontier(
        current_region, used_segments
    )
