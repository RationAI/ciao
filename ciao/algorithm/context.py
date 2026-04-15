from dataclasses import dataclass

import torch

from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor


@dataclass(frozen=True)
class SearchContext:
    """Immutable context object for the explanation hyperpixel search."""

    predictor: ModelPredictor
    input_batch: torch.Tensor
    replacement_image: torch.Tensor
    image_graph: ImageGraph
    target_class_idx: int
    seed_idx: int
    optimization_sign: int
    used_segments: frozenset[int]
    desired_length: int
    batch_size: int
