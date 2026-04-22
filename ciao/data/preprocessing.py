from pathlib import Path
from typing import cast

import torch
import torchvision.transforms as transforms
from PIL import Image


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


# ImageNet preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
)


def load_and_preprocess_image(
    image_path: str | Path, device: torch.device | None = None
) -> torch.Tensor:
    """Load and preprocess an image for the model.

    Args:
        image_path: Path to image file
        device: Device to place tensor on (defaults to cuda if available, else cpu)

    Returns:
        Preprocessed image tensor [3, 224, 224] on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use context manager to prevent file descriptor leaks
    with Image.open(image_path) as img:
        image = img.convert("RGB")
        tensor = cast("torch.Tensor", preprocess(image))  # (3, 224, 224)
        input_tensor = tensor.to(device)

    return input_tensor
