"""Graph utilities for segment manipulation using frozenset operations."""

import random
from collections.abc import Set
from dataclasses import dataclass

import torch


@dataclass
class ImageGraph:
    """Graph representation of image segments and their adjacencies."""

    segments: torch.Tensor
    adj_list: list[frozenset[int]]

    @property
    def num_segments(self) -> int:
        return len(self.adj_list)

    def get_frontier(
        self,
        current_segments: Set[int],
        used_segments: Set[int],
    ) -> frozenset[int]:
        """Compute the expansion frontier (valid neighbors) for graph traversal.

        The frontier is the set of segments adjacent to the current structure
        that can be added in the next step.

        A segment is in the frontier if:
        - It is adjacent to at least one segment in the current segments
        - It is NOT already in the current segments
        - It is NOT in the used_segments (respects global exclusion constraints)

        Args:
            current_segments: Set of current segment IDs
            used_segments: Set of globally excluded segments

        Returns:
            Frozenset of valid frontier segments
        """
        neighbors: set[int] = set()
        for node_id in current_segments:
            neighbors |= self.adj_list[node_id]

        # Remove segments already in the current segments and used segments
        frontier = frozenset(neighbors - current_segments - used_segments)
        return frontier

    def sample_connected_superset(
        self,
        base_segments: Set[int],
        target_length: int,
        used_segments: Set[int],
    ) -> frozenset[int]:
        """Simulates a random walk to build a full hyperpixel.

        Args:
            base_segments: Starting segments
            target_length: Desired number of segments in the final set
            used_segments: Segments to avoid

        Returns:
            Frozenset representing the connected superset
        """
        current_superset = set(base_segments)

        while len(current_superset) < target_length:
            frontier = self.get_frontier(current_superset, used_segments)
            if not frontier:
                break  # Dead end, cannot expand further

            # Pick a random neighbor and add it
            chosen = random.choice(tuple(frontier))
            current_superset.add(chosen)

        return frozenset(current_superset)
