from typing import Optional


class MCTSNode:
    def __init__(
        self, mask: int, parent: Optional["MCTSNode"] = None, prior_score: float = 0.0
    ):
        self.mask = mask
        self.parent = parent
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.max_value = -float("inf")

        self.rave_visits = 0
        self.rave_value_sum = 0.0
        self.rave_max_value = -float("inf")

        self.pending = 0  # virtual loss counter

        # RAVE-specific: Global RAVE prior for smart FPU initialization
        self.prior_score = prior_score
        self.frontier_cache: int | None = None

    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def rave_mean(self) -> float:
        return self.rave_value_sum / self.rave_visits if self.rave_visits > 0 else 0.0


class MCGSNode:
    def __init__(self, mask: int):
        self.mask = mask
        self.children: dict[int, MCGSNode] = {}  # segment_id -> child node

        self.edge_stats: dict[
            int, dict[str, float]
        ] = {}  # segment_id -> {'N': 0, 'W': 0.0, 'Q': 0.0, 'max_reward': -inf}
        self.rave_stats: dict[int, dict[str, float]] = {}
        self.pending_edges: dict[int, int] = {}  # segment_id -> pending count

        self.visits = 0
        self.value_sum = 0.0
        self.max_value = -float("inf")
        self.pending = 0  # virtual loss counter

    def init_edge(self, action: int) -> None:
        if action not in self.edge_stats:
            self.edge_stats[action] = {
                "N": 0,
                "W": 0.0,
                "Q": 0.0,
                "max_reward": -float("inf"),
            }
        if action not in self.rave_stats:
            self.rave_stats[action] = {
                "N": 0,
                "W": 0.0,
                "Q": 0.0,
                "max_reward": -float("inf"),
            }
        if action not in self.pending_edges:
            self.pending_edges[action] = 0
