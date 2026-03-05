"""Simple image path loading utilities."""

from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig


# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_image_loader(config: DictConfig) -> Iterator[Path]:
    """Create image loader based on configuration.

    Args:
        config: Hydra config object

    Returns:
        Iterator of Path objects

    Raises:
        ValueError: If neither image_path nor batch_path is specified
        FileNotFoundError: If single image_path does not exist
    """
    if config.data.get("image_path"):
        # Single image mode - validate file exists
        image_path = Path(config.data.image_path)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image_path must be a valid file, got: {image_path}. "
                "Check for typos or incorrect path configuration."
            )
        yield image_path

    elif config.data.get("batch_path"):
        # Directory mode - find all images with supported extensions
        directory = Path(config.data.batch_path)
        if not directory.is_dir():
            raise ValueError(
                f"batch_path must be a valid directory, got: {directory}. "
                "Check for typos or incorrect path configuration."
            )

        # Single rglob pass with suffix filtering
        for path in directory.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path

    else:
        raise ValueError("Must specify either image_path or batch_path in config")
