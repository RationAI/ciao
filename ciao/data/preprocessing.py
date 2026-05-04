from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from ciao.data.constants import IMAGENET_MEAN, IMAGENET_STD


# ImageNet preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ]
)

# PCam (resnet50.tiatoolbox-pcam) preprocessing — 96x96, no normalization
preprocess_pcam = transforms.Compose(
    [
        transforms.Resize(96, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
    ]
)


def _load_and_preprocess(
    image_path: str | Path,
    transform: Callable[..., torch.Tensor],
    device: torch.device | None = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with Image.open(image_path) as img:
        image = img.convert("RGB")
        return transform(image).to(device)


def load_and_preprocess_image(
    image_path: str | Path, device: torch.device | None = None
) -> torch.Tensor:
    """Load and preprocess an image with ImageNet normalization (224x224)."""
    return _load_and_preprocess(image_path, preprocess, device)


def load_and_preprocess_image_pcam(
    image_path: str | Path, device: torch.device | None = None
) -> torch.Tensor:
    """Load and preprocess an image for the PCam model (96x96, no normalization)."""
    return _load_and_preprocess(image_path, preprocess_pcam, device)
