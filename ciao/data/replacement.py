"""Image replacement strategies for masking operations."""

from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

from ciao.explainer.methods import Replacement


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def calculate_image_mean_color(input_tensor: torch.Tensor) -> torch.Tensor:
    """Calculate image mean color using ImageNet normalization constants.

    Args:
        input_tensor: Input tensor [3, H, W] or [1, 3, H, W] (ImageNet normalized)

    Returns:
        Mean color tensor [3, 1, 1] (ImageNet normalized)
    """
    device = input_tensor.device

    # Add batch dimension if needed
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Move normalization constants to same device
    imagenet_mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device)
    imagenet_std = IMAGENET_STD.view(1, 3, 1, 1).to(device)

    # Unnormalize, calculate mean, then re-normalize
    unnormalized = (input_tensor * imagenet_std) + imagenet_mean
    mean_color = unnormalized.mean(dim=(2, 3), keepdim=True)
    normalized_mean = (mean_color - imagenet_mean) / imagenet_std

    return normalized_mean.squeeze(0)  # Remove batch dimension


@dataclass
class MeanColorReplacement(Replacement):
    """Configuration for mean color replacement strategy."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        mean_color = calculate_image_mean_color(image)
        return mean_color.expand(-1, height, width)


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

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        input_batch = image.unsqueeze(0)
        return TF.gaussian_blur(
            input_batch,
            kernel_size=list(self.kernel_size),
            sigma=list(self.sigma),
        ).squeeze(0)


@dataclass
class InterlacingReplacement(Replacement):
    """Configuration for interlacing replacement strategy."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        replacement_image = image.clone()
        even_row_indices = torch.arange(0, height, 2)
        even_col_indices = torch.arange(0, width, 2)
        replacement_image[:, :, even_col_indices] = torch.flip(
            replacement_image[:, :, even_col_indices], dims=[1]
        )
        replacement_image[:, even_row_indices, :] = torch.flip(
            replacement_image[:, even_row_indices, :], dims=[2]
        )
        return replacement_image


@dataclass
class SolidColorReplacement(Replacement):
    """Configuration for solid color replacement strategy."""

    color: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        if len(self.color) != 3:
            raise ValueError(
                f"RGB color tuple must have exactly 3 elements, got {len(self.color)}"
            )
        if not all(0 <= c <= 255 for c in self.color):
            raise ValueError(
                f"RGB color values must be between 0 and 255, got {self.color}"
            )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        color_tensor = torch.tensor(
            self.color, dtype=image.dtype, device=image.device
        ).view(3, 1, 1)
        normalized_color = color_tensor / 255.0

        imagenet_mean = IMAGENET_MEAN.view(3, 1, 1).to(image.device)
        imagenet_std = IMAGENET_STD.view(3, 1, 1).to(image.device)

        normalized_color = (normalized_color - imagenet_mean) / imagenet_std
        return normalized_color.expand(-1, height, width)
