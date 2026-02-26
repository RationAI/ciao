"""Simple image path loading utilities."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def get_image_loader(config: Any) -> Iterator[Path]:
    """Create image loader based on configuration.

    Args:
        config: Hydra config object

    Returns:
        Iterator of Path objects

    Raises:
        ValueError: If neither image_path nor batch_path is specified
    """
    if config.data.get("image_path"):
        # Single image mode
        yield Path(config.data.image_path)

    elif config.data.get("batch_path"):
        # Directory mode
        directory = Path(config.data.batch_path)
        if not directory.is_dir():
            raise ValueError(
                f"batch_path must be a valid directory, got: {directory}. "
                "Check for typos or incorrect path configuration."
            )
        extensions = config.data.get(
            "image_extensions", [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        )
        for ext in extensions:
            yield from directory.glob(f"**/*{ext}")

    else:
        raise ValueError("Must specify either image_path or batch_path in config")
