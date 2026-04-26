"""Node classes for MCGS search structure."""

import math
from dataclasses import dataclass


@dataclass
class EdgeStats:
    """Per-edge (parent, action) statistics for MCGS."""

    visits: int = 0
    mean_value: float = 0.0
    max_value: float = -math.inf


class MCGSNode:
    """Node in a Monte Carlo Graph Search DAG.

    The same MCGSNode can be reached from multiple parents
    (a state shared via a transposition table). No ``parent`` pointer is
    stored; edge statistics live on the parent under ``edge_stats[action]``.
    """

    def __init__(self, region: frozenset[int]):
        self.region = region
        self.children: dict[int, MCGSNode] = {}
        self.edge_stats: dict[int, EdgeStats] = {}

        self.visits = 0
        self.mean_value = 0.0
        self.max_value = -math.inf
