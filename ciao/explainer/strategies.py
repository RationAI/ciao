"""Dataclasses for explanation methods and replacement strategies."""

from dataclasses import dataclass


@dataclass
class ExplanationMethod:
    """Base class for all hyperpixel building methods."""


@dataclass
class LookaheadMethod(ExplanationMethod):
    """Configuration for the lookahead hyperpixel building method."""

    lookahead_distance: int = 2


@dataclass
class Replacement:
    """Base class for image masking strategies."""


@dataclass
class MeanColorReplacement(Replacement):
    """Configuration for mean color replacement strategy."""


@dataclass
class BlurReplacement(Replacement):
    """Configuration for blur replacement strategy."""

    sigma: tuple[float, float] = (5.0, 5.0)
    kernel_size: tuple[int, int] = (15, 15)


@dataclass
class InterlacingReplacement(Replacement):
    """Configuration for interlacing replacement strategy."""


@dataclass
class SolidColorReplacement(Replacement):
    """Configuration for solid color replacement strategy."""

    color: tuple[int, int, int] = (0, 0, 0)


@dataclass
class SegmentationMethod:
    """Base class for image segmentation strategies."""


@dataclass
class HexagonalSegmentation(SegmentationMethod):
    """Configuration for hexagonal grid segmentation."""

    hex_radius: int = 4


@dataclass
class SquareSegmentation(SegmentationMethod):
    """Configuration for square grid segmentation."""

    square_size: int = 4
    neighborhood: int = 8
