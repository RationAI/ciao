"""Segmentation-based evaluation metrics for CIAO explanations."""

import torch

from ciao.data.imagenet_s import ImageNetSMapping, get_object_mask
from ciao.explainer.ciao_explainer import ExplanationResult


def compute_iou(
    result: ExplanationResult,
    gt_mask: torch.Tensor,
    mapping: ImageNetSMapping,
) -> float | None:
    """Compute IoU between the explanation's union region mask and the GT segmentation.

    The prediction mask is the union of all RegionResult regions, binarized over
    the explanation's segment map. The GT mask is resized to the explanation
    resolution (224x224) with nearest-neighbor interpolation.

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

    segment_ids = torch.tensor(list(union_segments), dtype=result.segments.dtype)
    pred_mask = torch.isin(result.segments, segment_ids)  # [H, W] bool

    object_mask = get_object_mask(gt_mask, result.target_class_idx, mapping)
    if object_mask is None:
        return None

    H, W = result.segments.shape
    gt_resized = (
        torch.nn.functional.interpolate(
            object_mask.float().unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        .bool()
    )

    intersection = int((pred_mask & gt_resized).sum())
    union = int((pred_mask | gt_resized).sum())
    if union == 0:
        return None
    return intersection / union
