from dataclasses import dataclass

import torch

from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor


@dataclass(frozen=True)
class SearchContext:
    """Immutable context object for the explanation region search."""

    predictor: ModelPredictor
    input_batch: torch.Tensor
    replacement_image: torch.Tensor
    image_graph: ImageGraph
    target_class_idx: int
    original_log_odds: torch.Tensor
    seed_idx: int
    optimization_sign: int
    used_segments: frozenset[int]
    desired_length: int
    batch_size: int

    def __post_init__(self) -> None:
        if self.desired_length < 1:
            raise ValueError(f"desired_length must be >= 1, got {self.desired_length}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.optimization_sign not in (1, -1):
            raise ValueError(
                f"optimization_sign must be 1 or -1, got {self.optimization_sign}"
            )
        if not (0 <= self.seed_idx < self.image_graph.num_segments):
            raise ValueError(
                f"seed_idx {self.seed_idx} is out of bounds (0 to {self.image_graph.num_segments - 1})"
            )
        if self.seed_idx in self.used_segments:
            raise ValueError(f"seed_idx {self.seed_idx} is already in used_segments")
