"""Simple image path resolution utilities."""

from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig


# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_image_paths(config: DictConfig) -> Iterator[Path]:
    """Generate paths to images based on configuration.

    Args:
        config: Hydra config object containing data.image_path or data.batch_path

    Returns:
        Iterator of Path objects pointing to valid images

    Raises:
        ValueError: If config specifies both or neither paths
        FileNotFoundError: If single image_path does not exist
        NotADirectoryError: If batch_path directory does not exist
    """
    image_path_value = config.data.get("image_path")
    batch_path_value = config.data.get("batch_path")

    if image_path_value and batch_path_value:
        raise ValueError("Specify exactly one of image_path or batch_path in config")

    if not image_path_value and not batch_path_value:
        raise ValueError("Must specify either image_path or batch_path in config")

    if image_path_value:
        image_path = Path(image_path_value)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"image_path must be a valid file, got: {image_path}. "
                "Check for typos or incorrect path configuration."
            )

    if batch_path_value:
        directory = Path(batch_path_value)
        if not directory.is_dir():
            raise NotADirectoryError(
                f"batch_path must be a valid directory, got: {directory}. "
                "Check for typos or incorrect path configuration."
            )

    if image_path_value:
        yield Path(image_path_value)
    else:
        for path in Path(batch_path_value).rglob("*"):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                yield path
