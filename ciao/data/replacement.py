"""Image replacement strategies for masking operations."""

import torch
import torchvision.transforms.functional as TF

from ciao.typing import ReplacementFn


# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def calculate_image_mean_color(input_tensor: torch.Tensor) -> torch.Tensor:
    """Calculate image mean color using ImageNet normalization constants.

    Args:
        input_tensor: Input tensor [3, H, W] (ImageNet normalized)

    Returns:
        Mean color tensor [3, 1, 1] (ImageNet normalized)
    """
    device = input_tensor.device

    # Move normalization constants to same device
    imagenet_mean = IMAGENET_MEAN.view(3, 1, 1).to(device)
    imagenet_std = IMAGENET_STD.view(3, 1, 1).to(device)

    # Unnormalize, calculate mean, then re-normalize
    unnormalized = (input_tensor * imagenet_std) + imagenet_mean
    mean_color = unnormalized.mean(dim=(1, 2), keepdim=True)
    normalized_mean = (mean_color - imagenet_mean) / imagenet_std

    return normalized_mean


def mean_color_replacement(image: torch.Tensor) -> torch.Tensor:
    """Mean color replacement strategy.

    Replaces an image by replacing everything with the global mean color.

    Args:
        image: Original input tensor of shape (3, H, W).

    Returns:
        torch.Tensor: Tensor containing just the mean color painted across all pixels.
    """
    _, height, width = image.shape
    mean_color = calculate_image_mean_color(image)
    return mean_color.expand(-1, height, width)


def make_blur_replacement(
    sigma: tuple[float, float] = (5.0, 5.0), kernel_size: tuple[int, int] = (15, 15)
) -> ReplacementFn:
    """Return a function that replaces image regions by applying gaussian blur.

    Args:
        sigma: X and Y Standard deviation of the Gaussian filter.
        kernel_size: X and Y size of the Gaussian blur kernel.

    Returns:
        ReplacementFn: A callable that generates a blurred image tensor.
    """
    # validation
    if any(s <= 0 for s in sigma):
        raise ValueError(f"sigma values must be > 0, got {sigma}")
    if any(k <= 0 or k % 2 == 0 for k in kernel_size):
        raise ValueError(
            f"kernel_size must be positive odd integers, got {kernel_size}"
        )

    def blur(image: torch.Tensor) -> torch.Tensor:
        input_batch = image.unsqueeze(0)
        return TF.gaussian_blur(
            input_batch,
            kernel_size=list(kernel_size),
            sigma=list(sigma),
        ).squeeze(0)

    return blur


def interlacing_replacement(image: torch.Tensor) -> torch.Tensor:
    """Interlacing replacement strategy.

    Replaces an image by interlacing pixels spatially. Flips alternating
    rows/columns to disrupt feature locality.

    Args:
        image: Original input tensor.

    Returns:
        torch.Tensor: Structurally scrambled interlaced image.
    """
    _, height, width = image.shape
    replacement_image = image.clone()
    device = image.device

    even_row_indices = torch.arange(0, height, 2, device=device)
    even_col_indices = torch.arange(0, width, 2, device=device)

    replacement_image[:, :, even_col_indices] = torch.flip(
        replacement_image[:, :, even_col_indices], dims=[1]
    )
    replacement_image[:, even_row_indices, :] = torch.flip(
        replacement_image[:, even_row_indices, :], dims=[2]
    )
    return replacement_image


def make_solid_color_replacement(
    color: tuple[int, int, int] = (0, 0, 0),
) -> ReplacementFn:
    """Return a function that generates a solid-color blackout replacement mask.

    Args:
        color: Solid RGB int bounds.

    Returns:
        ReplacementFn: A callable outputting a solid RGB normalized color mask.
    """
    if len(color) != 3:
        raise ValueError(
            f"RGB color tuple must have exactly 3 elements, got {len(color)}"
        )
    if not all(0 <= c <= 255 for c in color):
        raise ValueError(f"RGB color values must be between 0 and 255, got {color}")

    def replacement(image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape

        color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(
            3, 1, 1
        )

        normalized_color = color_tensor / 255.0

        imagenet_mean = IMAGENET_MEAN.view(3, 1, 1).to(image.device)
        imagenet_std = IMAGENET_STD.view(3, 1, 1).to(image.device)

        normalized_color = (normalized_color - imagenet_mean) / imagenet_std
        return normalized_color.expand(-1, height, width)

    return replacement
