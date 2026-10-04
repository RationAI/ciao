"""ImageNet-S dataset utilities: class mapping, mask loading, image-mask iteration."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


_DATA_DIR = Path(__file__).parent
_CATEGORIES_FILE = _DATA_DIR / "ImageNetS_categories_im919.txt"
_CLASS_INDEX_FILE = _DATA_DIR / "imagenet_class_index.json"

# n04356056 (sunglasses) merged into n04355933 (sunglass), etc.
# Pixels from the key class are labeled with the value class in the dataset.
_MERGE_WNIDS: dict[str, str] = {
    "n04356056": "n04355933",
    "n04493381": "n02808440",
    "n03642806": "n03832673",
    "n04008634": "n03773504",
    "n03887697": "n15075141",
}


@dataclass(frozen=True)
class ImageNetSMapping:
    # imagenet_1000_idx (0-999) → imagenet_s_id (1-919); missing key = class not in dataset
    imagenet_idx_to_s_id: dict[int, int]
    # imagenet_s_id (1-919) → imagenet_1000_idx (0-999)
    s_id_to_imagenet_idx: dict[int, int]


@cache
def build_imagenet_s_mapping() -> ImageNetSMapping:
    """Build bidirectional mapping between ImageNet-1000 indices and ImageNet-S-919 class IDs.

    Cached after first call. Handles the 5 class merges defined in the dataset.
    """
    with open(_CATEGORIES_FILE) as f:
        wnids = sorted(line.strip() for line in f if line.strip())
    # imagenet_s_id is 1-indexed position in the sorted list (per dataset spec)
    s_id_to_wnid = {i + 1: wnid for i, wnid in enumerate(wnids)}
    wnid_to_s_id = {v: k for k, v in s_id_to_wnid.items()}

    with open(_CLASS_INDEX_FILE) as f:
        class_index = json.load(f)
    idx_to_wnid_1000: dict[int, str] = {int(k): v[0] for k, v in class_index.items()}

    imagenet_idx_to_s_id: dict[int, int] = {}
    s_id_to_imagenet_idx: dict[int, int] = {}

    for idx_1000, wnid in idx_to_wnid_1000.items():
        # Merged-away classes redirect to their merge target in the mask.
        effective_wnid = _MERGE_WNIDS.get(wnid, wnid)
        s_id = wnid_to_s_id.get(effective_wnid)
        if s_id is None:
            continue
        imagenet_idx_to_s_id[idx_1000] = s_id
        if wnid == effective_wnid:
            s_id_to_imagenet_idx[s_id] = idx_1000

    return ImageNetSMapping(
        imagenet_idx_to_s_id=imagenet_idx_to_s_id,
        s_id_to_imagenet_idx=s_id_to_imagenet_idx,
    )


def load_mask(path: Path) -> torch.Tensor:
    """Load an ImageNet-S segmentation PNG as a [H, W] int32 tensor of class IDs.

    0 = background/unlabeled, 1000 = ignored region.
    Encoding: class_id = R + G * 256.
    """
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.int32)
    class_ids = arr[:, :, 0] + arr[:, :, 1] * 256
    return torch.tensor(class_ids, dtype=torch.int32)


def get_object_mask(
    mask: torch.Tensor,
    target_class_idx: int,
    mapping: ImageNetSMapping,
) -> torch.Tensor | None:
    """Return a [H, W] bool tensor of pixels belonging to target_class_idx.

    Returns None if the target ImageNet-1000 class is not present in ImageNet-S-919.
    """
    s_id = mapping.imagenet_idx_to_s_id.get(target_class_idx)
    if s_id is None:
        return None
    return mask == s_id


def iter_image_mask_pairs(root: Path) -> Iterator[tuple[Path, Path]]:
    """Yield (image_path, mask_path) pairs from a root with images/ and masks/ subdirs."""
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images_dir = root / "images"
    masks_dir = root / "masks"
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        mask_path = masks_dir / (image_path.stem + ".png")
        if mask_path.exists():
            yield image_path, mask_path
