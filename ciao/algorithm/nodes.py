"""Node classes for MCTS search tree."""

import math


class MCTSNode:
    def __init__(self, region: frozenset[int]):
        self.region = region
        self.children: dict[int, MCTSNode] = {}

        self.visits = 0
        self.mean_value = 0.0
        self.max_value = -math.inf
