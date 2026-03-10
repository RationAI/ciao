"""Image replacement strategies for masking operations."""

import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt


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


def get_replacement_image(
    input_tensor: torch.Tensor,
    replacement: str = "mean_color",
    color: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """Generate replacement image for masking operations.

    Args:
        input_tensor: Input tensor [3, H, W] (ImageNet normalized)
        replacement: Strategy - "mean_color", "interlacing", "blur", or "solid_color"
        color: For solid_color mode, RGB tuple (0-255). Defaults to black (0, 0, 0)

    Returns:
        replacement_image: torch tensor [3, H, W] on same device
    """
    device = input_tensor.device

    # Extract spatial dimensions from input tensor
    _, height, width = input_tensor.shape

    if replacement == "mean_color":
        # Fill entire image with mean color
        mean_color = calculate_image_mean_color(input_tensor)  # [3, 1, 1]
        replacement_image = mean_color.expand(-1, height, width)  # [3, H, W]

    elif replacement == "interlacing":
        # Create interlaced pattern: even columns flipped vertically, then even rows flipped horizontally
        replacement_image = input_tensor.clone()
        even_row_indices = torch.arange(0, height, 2)  # Even row indices
        even_col_indices = torch.arange(0, width, 2)  # Even column indices

        # Step 1: Flip even columns vertically (upside down)
        replacement_image[:, :, even_col_indices] = torch.flip(
            replacement_image[:, :, even_col_indices], dims=[1]
        )

        # Step 2: Flip even rows horizontally (left-right)
        replacement_image[:, even_row_indices, :] = torch.flip(
            replacement_image[:, even_row_indices, :], dims=[2]
        )

    elif replacement == "blur":
        # Apply Gaussian blur using torchvision functional API
        input_batch = input_tensor.unsqueeze(0)  # [1, 3, H, W]
        replacement_image = TF.gaussian_blur(
            input_batch, kernel_size=[7, 7], sigma=[1.5, 1.5]
        ).squeeze(0)  # [3, H, W]

    elif replacement == "solid_color":
        # Fill with specified solid color (expects RGB values in 0-255 range)
        # Convert color to torch tensor
        color_tensor = torch.tensor(color, dtype=torch.float32, device=device)

        # Convert from 0-255 range to 0-1 range
        color_tensor = color_tensor / 255.0

        # Apply ImageNet normalization
        mean = IMAGENET_MEAN.view(3, 1, 1).to(device)
        std = IMAGENET_STD.view(3, 1, 1).to(device)
        normalized_color = (color_tensor.view(3, 1, 1) - mean) / std
        replacement_image = normalized_color.expand(-1, height, width)  # [3, H, W]

    else:
        raise ValueError(f"Unknown replacement strategy: {replacement}")

    return replacement_image


def plot_image_mean_color(input_tensor: torch.Tensor) -> None:
    """Display the mean color of the image.

    Args:
        input_tensor: Input tensor [3, H, W] (ImageNet normalized)

    Note:
        The visualization shows the normalized tensor (ImageNet normalization).
    """
    mean = IMAGENET_MEAN.view(3, 1, 1).to(
        device=input_tensor.device, dtype=input_tensor.dtype
    )
    std = IMAGENET_STD.view(3, 1, 1).to(
        device=input_tensor.device, dtype=input_tensor.dtype
    )
    normalized_mean = calculate_image_mean_color(input_tensor)
    display_mean = ((normalized_mean * std) + mean).clamp(0, 1)
    plt.imshow(display_mean.permute(1, 2, 0).detach().cpu())
    plt.show()
