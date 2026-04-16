"""Simple image path resolution utilities."""

import itertools
from collections.abc import Iterator
from pathlib import Path

from omegaconf import DictConfig


# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def iter_image_paths(config: DictConfig) -> Iterator[Path]:
    """Generate paths to images based on configuration.

    Reads ``config.data.image_path`` (single image) or
    ``config.data.batch_path`` (directory), with an optional
    ``config.data.limit`` to cap the number of yielded paths.

    Args:
        config: Hydra config object containing data.image_path or data.batch_path

    Returns:
        Iterator of Path objects pointing to valid images

    Raises:
        ValueError: If config specifies both or neither paths,
                    or if the image extension is unsupported.
        FileNotFoundError: If single image_path does not exist.
        NotADirectoryError: If batch_path directory does not exist.
    """
    image_path_value = config.data.get("image_path")
    batch_path_value = config.data.get("batch_path")
    limit: int | None = config.data.get("limit")

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
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"image_path must use a supported image extension, got: {image_path}"
            )
        yield image_path
        return

    directory = Path(batch_path_value)
    if not directory.is_dir():
        raise NotADirectoryError(
            f"batch_path must be a valid directory, got: {directory}. "
            "Check for typos or incorrect path configuration."
        )

    paths = (
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    yield from itertools.islice(paths, limit)
