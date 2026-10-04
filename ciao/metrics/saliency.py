"""Saliency map construction and saliency-based evaluation metrics for CIAO explanations."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from ciao.data.imagenet_s import ImageNetSMapping, get_object_mask
from ciao.model.predictor import ModelPredictor


def build_saliency_map(
    binary_masks: list[np.ndarray],
    sigma_fraction: float = 0.09,
) -> np.ndarray:
    """Aggregate binary masks into a saliency map using the extremal perturbations technique.

    Sums the binary masks and applies a Gaussian filter with sigma equal to
    ``sigma_fraction`` of the shorter image side, then normalises to [0, 1].

    Args:
        binary_masks: List of [H, W] boolean or float arrays (one per run).
        sigma_fraction: Fraction of the shorter image side used as Gaussian sigma.

    Returns:
        [H, W] float32 saliency map normalised to [0, 1].
    """
    if not binary_masks:
        raise ValueError("binary_masks must not be empty")

    summed = np.stack(binary_masks).sum(axis=0).astype(np.float32)
    sigma = sigma_fraction * min(summed.shape)
    smoothed = gaussian_filter(summed, sigma=sigma)
    max_val = float(smoothed.max())
    return smoothed / max_val if max_val > 0 else smoothed


def compute_deletion_curve(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int,
    predictor: ModelPredictor,
    replacement_image: torch.Tensor,
    n_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the deletion curve (fractions, probabilities).

    Progressively masks the highest-saliency pixels (replacing them with
    ``replacement_image``) and measures the target class probability at
    ``n_steps + 1`` equally spaced fractions from 0% to 100% masked.

    Args:
        image: [1, C, H, W] preprocessed input tensor.
        saliency_map: [H, W] float saliency map (higher = more important).
        target_class_idx: Index of the class being explained.
        predictor: ModelPredictor used for inference.
        replacement_image: [C, H, W] replacement tensor (same as used during explanation).
        n_steps: Number of masking steps between 0 and 100 % (inclusive endpoints).

    Returns:
        Tuple of (fractions, probs), each a 1-D float64 array of length n_steps + 1.
    """
    H, W = saliency_map.shape
    n_pixels = H * W
    flat_order = np.argsort(saliency_map.ravel())[::-1]  # highest saliency first

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    probs: list[float] = []

    device = predictor.device
    input_flat = image.squeeze(0).view(3, n_pixels).to(device)  # [C, H*W]
    repl_flat = replacement_image.view(3, n_pixels).to(device)  # [C, H*W]

    for frac in fractions:
        n_masked = round(frac * n_pixels)
        mask = torch.zeros(n_pixels, dtype=torch.bool, device=device)
        if n_masked > 0:
            indices = torch.tensor(
                flat_order[:n_masked].copy(), dtype=torch.long, device=device
            )
            mask[indices] = True

        masked_flat = torch.where(mask.unsqueeze(0), repl_flat, input_flat)
        masked_image = masked_flat.view(image.shape[1:]).unsqueeze(0)

        prob = float(
            predictor.get_predictions(masked_image)[0, target_class_idx].item()
        )
        probs.append(prob)

    return fractions, np.array(probs, dtype=np.float64)


def compute_deletion_auc(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int,
    predictor: ModelPredictor,
    replacement_image: torch.Tensor,
    n_steps: int = 10,
) -> float:
    """Compute the deletion AUC metric (area under the deletion curve).

    Lower AUC means the saliency map correctly identified important regions
    (masking them quickly destroys the prediction).

    Args:
        image: [1, C, H, W] preprocessed input tensor.
        saliency_map: [H, W] float saliency map (higher = more important).
        target_class_idx: Index of the class being explained.
        predictor: ModelPredictor used for inference.
        replacement_image: [C, H, W] replacement tensor (same as used during explanation).
        n_steps: Number of masking steps between 0 and 100 % (inclusive endpoints).

    Returns:
        Scalar AUC in [0, 1].
    """
    fractions, probs = compute_deletion_curve(
        image, saliency_map, target_class_idx, predictor, replacement_image, n_steps
    )
    return float(np.trapezoid(probs, fractions))


def compute_pointing_game(
    saliency_map: np.ndarray,
    target_class_idx: int,
    gt_mask: torch.Tensor,
    mapping: ImageNetSMapping,
) -> bool | None:
    """Compute the pointing game metric for a single image.

    Checks whether the pixel with the highest saliency value falls inside the
    ground truth object region for ``target_class_idx``.  The GT mask is
    resized to the saliency map resolution with nearest-neighbour interpolation.

    Args:
        saliency_map: [H, W] float saliency map (higher = more important).
        target_class_idx: ImageNet-1000 class index being explained.
        gt_mask: [H', W'] int32 tensor from load_mask() at original resolution.
        mapping: ImageNetSMapping from build_imagenet_s_mapping().

    Returns:
        True (hit) / False (miss), or None if the class is not in ImageNet-S-919.
    """
    object_mask = get_object_mask(gt_mask, target_class_idx, mapping)
    if object_mask is None:
        return None

    H, W = saliency_map.shape
    gt_resized = (
        torch.nn.functional.interpolate(
            object_mask.float().unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="nearest",
        )
        .squeeze()
        .bool()
        .numpy()
    )

    peak = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    return bool(gt_resized[peak])


def compute_insertion_curve(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int,
    predictor: ModelPredictor,
    replacement_image: torch.Tensor,
    n_steps: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the insertion curve (fractions, probabilities).

    Starts from a fully replaced image and progressively reveals the
    highest-saliency pixels from the original image, measuring the target
    class probability at ``n_steps + 1`` equally spaced fractions.

    Args:
        image: [1, C, H, W] preprocessed input tensor.
        saliency_map: [H, W] float saliency map (higher = more important).
        target_class_idx: Index of the class being explained.
        predictor: ModelPredictor used for inference.
        replacement_image: [C, H, W] replacement tensor (same as used during explanation).
        n_steps: Number of reveal steps between 0 and 100 % (inclusive endpoints).

    Returns:
        Tuple of (fractions, probs), each a 1-D float64 array of length n_steps + 1.
    """
    H, W = saliency_map.shape
    n_pixels = H * W
    flat_order = np.argsort(saliency_map.ravel())[::-1]  # highest saliency first

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    probs: list[float] = []

    device = predictor.device
    input_flat = image.squeeze(0).view(3, n_pixels).to(device)  # [C, H*W]
    repl_flat = replacement_image.view(3, n_pixels).to(device)  # [C, H*W]

    for frac in fractions:
        n_revealed = round(frac * n_pixels)
        revealed = torch.zeros(n_pixels, dtype=torch.bool, device=device)
        if n_revealed > 0:
            indices = torch.tensor(
                flat_order[:n_revealed].copy(), dtype=torch.long, device=device
            )
            revealed[indices] = True

        blended_flat = torch.where(revealed.unsqueeze(0), input_flat, repl_flat)
        blended_image = blended_flat.view(image.shape[1:]).unsqueeze(0)

        prob = float(
            predictor.get_predictions(blended_image)[0, target_class_idx].item()
        )
        probs.append(prob)

    return fractions, np.array(probs, dtype=np.float64)


def compute_insertion_auc(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int,
    predictor: ModelPredictor,
    replacement_image: torch.Tensor,
    n_steps: int = 10,
) -> float:
    """Compute insertion AUC (area under the insertion curve). Higher is better.

    Args:
        image: [1, C, H, W] preprocessed input tensor.
        saliency_map: [H, W] float saliency map (higher = more important).
        target_class_idx: Index of the class being explained.
        predictor: ModelPredictor used for inference.
        replacement_image: [C, H, W] replacement tensor (same as used during explanation).
        n_steps: Number of reveal steps between 0 and 100 % (inclusive endpoints).

    Returns:
        Scalar AUC in [0, 1].
    """
    fractions, probs = compute_insertion_curve(
        image, saliency_map, target_class_idx, predictor, replacement_image, n_steps
    )
    return float(np.trapezoid(probs, fractions))


def compute_iou_top_fraction(
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
    *,
    top_fraction: float | None = None,
    top_k: int | None = None,
) -> float | None:
    """Compute IoU between the top-K saliency pixels and a binary ground truth mask.

    Exactly one of ``top_fraction`` or ``top_k`` must be provided.

    Args:
        saliency_map: [H, W] float array (higher = more salient).
        gt_mask: [H, W] bool or binary int array at the same resolution as saliency_map.
        top_fraction: Fraction of pixels to select (e.g., 0.1 for top 10%).
        top_k: Exact number of pixels to select.

    Returns:
        IoU in [0, 1], or None if either mask is empty.
    """
    if (top_fraction is None) == (top_k is None):
        raise ValueError("Exactly one of top_fraction or top_k must be provided.")

    H, W = saliency_map.shape
    n_pixels = H * W
    k = top_k if top_k is not None else max(1, round(top_fraction * n_pixels))
    k = min(k, n_pixels)

    flat = saliency_map.ravel()
    threshold_idx = np.argpartition(flat, -k)[-k:]
    pred_flat = np.zeros(n_pixels, dtype=bool)
    pred_flat[threshold_idx] = True
    pred_mask = pred_flat.reshape(H, W)

    gt = gt_mask.astype(bool)
    intersection = int((pred_mask & gt).sum())
    union = int((pred_mask | gt).sum())
    if union == 0:
        return None
    return intersection / union
