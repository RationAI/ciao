"""Segmentation-based evaluation metrics for CIAO explanations."""

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from ciao.data.imagenet_s import ImageNetSMapping, get_object_mask
from ciao.explainer.ciao_explainer import ExplanationResult


# Must match the Resize(256) step in ciao.data.preprocessing's preprocessing
# pipeline: the model sees a center-cropped view of a 256-resize, not a plain
# downscale, so the GT mask needs the same resize-then-crop to stay aligned.
_PREPROCESS_RESIZE_SIZE = 256


def compute_iou(
    result: ExplanationResult,
    gt_mask: torch.Tensor,
    mapping: ImageNetSMapping,
) -> float | None:
    """Compute IoU between the explanation's union region mask and the GT segmentation.

    The prediction mask is the union of all RegionResult regions, binarized over
    the explanation's segment map. The GT mask is aligned to the explanation
    resolution by replicating the exact Resize(256) + CenterCrop(224) transform
    applied to the model input (see ciao.data.preprocessing) - not by
    independently squashing it to (224, 224), which would misalign non-square
    images and pull in border content the model's crop actually discarded.
    Nearest-neighbor interpolation preserves discrete class labels.

    Args:
        result: ExplanationResult from CIAOExplainer.explain().
        gt_mask: [H', W'] int32 tensor from load_mask(), original image resolution.
        mapping: ImageNetSMapping from build_imagenet_s_mapping().

    Returns:
        IoU in [0, 1], or None if the target class is not in ImageNet-S-919
        or neither mask has any foreground pixels.
    """
    if not result.regions:
        return None

    union_segments = frozenset().union(*[r.region for r in result.regions])
    if not union_segments:
        return None

    segment_ids = torch.tensor(
        list(union_segments), dtype=result.segments.dtype, device=result.segments.device
    )
    pred_mask = torch.isin(result.segments, segment_ids).cpu()  # [H, W] bool, CPU

    object_mask = get_object_mask(gt_mask, result.target_class_idx, mapping)
    if object_mask is None:
        return None

    H, W = result.segments.shape
    mask_chw = object_mask.to(torch.uint8).unsqueeze(0)  # [1, H', W']
    resized = TF.resize(
        mask_chw, size=_PREPROCESS_RESIZE_SIZE, interpolation=InterpolationMode.NEAREST
    )
    gt_resized = TF.center_crop(resized, [H, W]).squeeze(0).bool()

    intersection = int((pred_mask & gt_resized).sum())
    union = int((pred_mask | gt_resized).sum())
    if union == 0:
        return None
    return intersection / union
