"""Node classes for MCTS search tree."""

import math
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
        self.max_value = -math.inf
