"""Saliency map construction and deletion metric for CIAO explanations."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

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


def compute_deletion_auc(
    image: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int,
    predictor: ModelPredictor,
    replacement_image: torch.Tensor,
    n_steps: int = 10,
) -> float:
    """Compute the deletion AUC metric.

    Progressively masks the highest-saliency pixels (replacing them with
    ``replacement_image``) and measures the target class probability at
    ``n_steps + 1`` equally spaced fractions from 0% to 100% masked.
    Returns the area under the resulting curve (trapezoidal rule).

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
    H, W = saliency_map.shape
    n_pixels = H * W
    flat_order = np.argsort(saliency_map.ravel())[::-1]  # highest saliency first

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    probs: list[float] = []

    device = predictor.device
    input_flat = image.squeeze(0).view(3, n_pixels).to(device)  # [C, H*W]
    repl_flat = replacement_image.view(3, n_pixels).to(device)  # [C, H*W]

    for frac in fractions:
        n_masked = int(round(frac * n_pixels))
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

    return float(np.trapezoid(probs, fractions))
