"""Node classes for MCGS and MCTS search structures."""

import math
from dataclasses import dataclass


@dataclass
class EdgeStats:
    """Per-edge (parent, action) visit counter for MCGS.

    Q values are read from the child node directly (recursive Q), so edges
    only need to track traversal counts for the UCT exploration term.
    """

    visits: int = 0


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

        # Direct-evaluation counters for the recursive Q formula.
        # Only updated when this node is the simulation leaf, not when traversed.
        self._own_visits: int = 0
        self._own_value_sum: float = 0.0


class MCTSNode:
    def __init__(self, region: frozenset[int]):
        self.region = region
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0
        self.mean_value = 0.0
        self.max_value = -math.inf
