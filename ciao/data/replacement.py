"""Image replacement strategies for masking operations."""

import torch
import torchvision.transforms.functional as TF

from ciao.data.constants import IMAGENET_MEAN, IMAGENET_STD
from ciao.typing import ReplacementFn


def make_mean_color_replacement(
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> ReplacementFn:
    """Return a replacement function that fills the image with its per-image mean color.

    Args:
        mean: Per-channel normalization mean used during preprocessing.
        std: Per-channel normalization std used during preprocessing.
    """
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            f"mean and std must each have 3 elements, got {len(mean)} and {len(std)}"
        )
    if any(s == 0 for s in std):
        raise ValueError(f"std values must be non-zero, got {std}")
    t_mean_cpu = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    t_std_cpu = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def replacement(image: torch.Tensor) -> torch.Tensor:
        t_mean = t_mean_cpu.to(device=image.device, dtype=image.dtype)
        t_std = t_std_cpu.to(device=image.device, dtype=image.dtype)
        unnormalized = (image * t_std) + t_mean
        mean_color = unnormalized.mean(dim=(1, 2), keepdim=True)
        normalized_mean = (mean_color - t_mean) / t_std
        _, height, width = image.shape
        return normalized_mean.expand(-1, height, width)

    return replacement


def imagenet_mean_replacement(image: torch.Tensor) -> torch.Tensor:
    """Replace the image with the ImageNet dataset mean (zeros in normalized space).

    Args:
        image: Original input tensor of shape (3, H, W).

    Returns:
        Tensor filled with the ImageNet mean in normalized space.
    """
    _, height, width = image.shape
    return torch.zeros((3, 1, 1), device=image.device, dtype=image.dtype).expand(
        -1, height, width
    )


def imagenet_mean_replacement(image: torch.Tensor) -> torch.Tensor:
    """ImageNet mean replacement strategy.

    Replaces an image by replacing everything with the dataset-level
    ImageNet mean color. Assumes the input is already ImageNet-normalized,
    under which the dataset mean maps to the zero tensor.

    Args:
        image: ImageNet-normalized input tensor of shape (3, H, W).

    Returns:
        torch.Tensor: Zero tensor of the same shape, dtype, and device as input.
    """
    return torch.zeros_like(image)


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
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> ReplacementFn:
    """Return a function that generates a solid-color replacement image.

    Args:
        color: Solid RGB values in [0, 255].
        mean: Per-channel normalization mean used during preprocessing.
        std: Per-channel normalization std used during preprocessing.

    Returns:
        ReplacementFn: A callable outputting a solid RGB normalized color mask.
    """
    if len(color) != 3:
        raise ValueError(
            f"RGB color tuple must have exactly 3 elements, got {len(color)}"
        )
    if not all(0 <= c <= 255 for c in color):
        raise ValueError(f"RGB color values must be between 0 and 255, got {color}")
    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            f"mean and std must each have 3 elements, got {len(mean)} and {len(std)}"
        )
    if any(s == 0 for s in std):
        raise ValueError(f"std values must be non-zero, got {std}")

    color_tensor = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
    t_mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    t_std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    normalized_color_cpu = (color_tensor / 255.0 - t_mean) / t_std

    def replacement(image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        normalized_color = normalized_color_cpu.to(
            device=image.device, dtype=image.dtype
        )
        return normalized_color.expand(-1, height, width)

    return replacement
