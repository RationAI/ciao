"""Node classes for MCTS search tree."""

from typing import Optional


class MCTSNode:
    def __init__(
        self,
        region: frozenset[int],
        parent: Optional["MCTSNode"] = None,
    ):
        self.region = region
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}

        self.visits = 0
        self.mean_value = 0.0
        self.pending = 0  # virtual loss counter
