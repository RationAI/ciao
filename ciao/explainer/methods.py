"""Base interfaces and implementations for explanation methods."""

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
            ctx=ctx,
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
