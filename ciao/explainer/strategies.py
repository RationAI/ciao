"""Dataclasses for explanation methods, replacement strategies, and segmentation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ciao.algorithm.context import SearchContext
from ciao.algorithm.graph import ImageGraph
from ciao.scoring.hyperpixel import HyperpixelResult


class ExplanationMethod(ABC):
    """Base class for all hyperpixel building methods."""

    @abstractmethod
    def __call__(self, ctx: SearchContext) -> HyperpixelResult: ...


@dataclass
class LookaheadMethod(ExplanationMethod):
    """Lookahead hyperpixel building strategy."""

    lookahead_distance: int = 2

    def __post_init__(self) -> None:
        if self.lookahead_distance < 0:
            raise ValueError(
                f"lookahead_distance must be >= 0, got {self.lookahead_distance}"
            )

    def __call__(self, ctx: SearchContext) -> HyperpixelResult:
        from ciao.algorithm.lookahead import build_hyperpixel_greedy_lookahead

        return build_hyperpixel_greedy_lookahead(
            predictor=ctx.predictor,
            input_batch=ctx.input_batch,
            replacement_image=ctx.replacement_image,
            image_graph=ctx.image_graph,
            target_class_idx=ctx.target_class_idx,
            seed_idx=ctx.seed_idx,
            optimization_sign=ctx.optimization_sign,
            used_segments=set(ctx.used_segments),
            desired_length=ctx.desired_length,
            batch_size=ctx.batch_size,
            lookahead_distance=self.lookahead_distance,
        )


class Replacement(ABC):
    """Base class for image masking strategies."""

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor: ...


class SegmentationMethod(ABC):
    """Base class for image segmentation strategies."""

    @abstractmethod
    def __call__(self, image: torch.Tensor) -> ImageGraph: ...
