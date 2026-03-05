"""Data loading utilities for CIAO."""

from ciao.data.loader import get_image_loader
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.data.replacement import (
    calculate_image_mean_color,
    get_replacement_image,
    plot_image_mean_color,
)
from ciao.data.segmentation import create_segmentation


__all__ = [
    "calculate_image_mean_color",
    "create_segmentation",
    "get_image_loader",
    "get_replacement_image",
    "load_and_preprocess_image",
    "plot_image_mean_color",
]
