from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image


# ImageNet preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).to(device)  # (3, 224, 224) - on correct device

    return input_tensor
