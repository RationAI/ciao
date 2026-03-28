"""Dataclasses for explanation methods, replacement strategies, and segmentation strategies."""

from dataclasses import dataclass


@dataclass
class ExplanationMethod:
    """Base class for all hyperpixel building methods."""


@dataclass
class LookaheadMethod(ExplanationMethod):
    """Configuration for the lookahead hyperpixel building method."""

    lookahead_distance: int = 2

    def __post_init__(self) -> None:
        if self.lookahead_distance < 0:
            raise ValueError(
                f"lookahead_distance must be >= 0, got {self.lookahead_distance}"
            )


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

    def __post_init__(self) -> None:
        for s in self.sigma:
            if s <= 0:
                raise ValueError(f"sigma values must be > 0, got {self.sigma}")
        for k in self.kernel_size:
            if k <= 0 or k % 2 == 0:
                raise ValueError(
                    f"kernel_size values must be positive odd integers, got {self.kernel_size}"
                )


@dataclass
class InterlacingReplacement(Replacement):
    """Configuration for interlacing replacement strategy."""


@dataclass
class SolidColorReplacement(Replacement):
    """Configuration for solid color replacement strategy."""

    color: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if not all(0 <= c <= 255 for c in self.color):
            raise ValueError(
                f"RGB color values must be between 0 and 255, got {self.color}"
            )


@dataclass
class SegmentationMethod:
    """Base class for image segmentation strategies."""


@dataclass
class HexagonalSegmentation(SegmentationMethod):
    """Configuration for hexagonal grid segmentation."""

    hex_radius: int = 4

    def __post_init__(self) -> None:
        if self.hex_radius <= 0:
            raise ValueError(f"hex_radius must be > 0, got {self.hex_radius}")


@dataclass
class SquareSegmentation(SegmentationMethod):
    """Configuration for square grid segmentation."""

    square_size: int = 4
    neighborhood: int = 8

    def __post_init__(self) -> None:
        if self.square_size <= 0:
            raise ValueError(f"square_size must be > 0, got {self.square_size}")
        if self.neighborhood <= 0:
            raise ValueError(f"neighborhood must be > 0, got {self.neighborhood}")
