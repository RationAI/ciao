from collections.abc import Sequence, Set
from dataclasses import dataclass

import torch

from ciao.model.predictor import ModelPredictor


@dataclass
class HyperpixelResult:
    """Type definition for the output of hyperpixel building algorithms."""

    region: frozenset[int]
    score: float


def _prepare_tensors_for_model(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    replacement_image: torch.Tensor,
    segments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move tensors to the predictor's device and align dtypes."""
    input_batch = input_batch.to(predictor.device)
    replacement_image = replacement_image.to(
        device=predictor.device, dtype=input_batch.dtype
    )
    gpu_segments = segments.to(predictor.device)
    return input_batch, replacement_image, gpu_segments


def _build_mask_tensor(
    gpu_segments: torch.Tensor,
    segment_ids_slice: Sequence[Set[int]],
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean mask tensor [batch, H, W] from segment ID sets."""
    mask_list = []
    for segment_ids in segment_ids_slice:
        target_ids = torch.tensor(
            list(segment_ids), dtype=gpu_segments.dtype, device=device
        )
        mask_list.append(torch.isin(gpu_segments, target_ids))
    return torch.stack(mask_list)


def _apply_masks(
    input_batch: torch.Tensor,
    mask_tensor: torch.Tensor,
    replacement_image: torch.Tensor,
) -> torch.Tensor:
    """Apply boolean masks to replicate+replace input images in one broadcast op."""
    current_batch_size = mask_tensor.shape[0]
    batch_inputs = input_batch.repeat(current_batch_size, 1, 1, 1)
    return torch.where(
        mask_tensor.unsqueeze(1),  # [batch, 1, H, W]
        replacement_image.unsqueeze(0),  # [1, C, H, W]
        batch_inputs,  # [batch, C, H, W]
    )


def _compute_batch_deltas(
    predictor: ModelPredictor,
    batch_inputs: torch.Tensor,
    original_logit: torch.Tensor,
    target_class_idx: int,
) -> list[float]:
    """Run inference and return per-sample delta scores."""
    masked_logits = predictor.get_class_logit_batch(batch_inputs, target_class_idx)
    deltas_tensor = original_logit - masked_logits
    return deltas_tensor.tolist()


def calculate_hyperpixel_deltas(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    segment_sets: Sequence[Set[int]],
    replacement_image: torch.Tensor,
    target_class_idx: int,
    batch_size: int = 64,
) -> list[float]:
    """Calculate masking deltas for hyperpixel candidates using batched inference.

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch [1, C, H, W]
        segments: Pixel-to-segment mapping array [H, W]
        segment_sets: List of segment ID sets, e.g. [{1,2,3}, {4,5,6}]
        replacement_image: Replacement tensor [C, H, W]
        target_class_idx: Target class index
        batch_size: Batch size for internal batching

    Returns:
        Delta scores for each candidate
    """
    if not segment_sets:
        return []

    for i, segment_ids in enumerate(segment_sets):
        if not segment_ids:
            raise ValueError(f"Empty segment set at index {i}")

    input_batch, replacement_image, gpu_segments = _prepare_tensors_for_model(
        predictor, input_batch, replacement_image, segments
    )

    with torch.no_grad():
        # Keep original_logit as a tensor to maintain vectorized operations
        original_logit = predictor.get_class_logit_batch(input_batch, target_class_idx)[
            0
        ]

        all_deltas: list[float] = []
        num_masks = len(segment_sets)

        for batch_start in range(0, num_masks, batch_size):
            batch_end = min(batch_start + batch_size, num_masks)
            segment_slice = segment_sets[batch_start:batch_end]

            mask_tensor = _build_mask_tensor(
                gpu_segments, segment_slice, predictor.device
            )
            batch_inputs = _apply_masks(input_batch, mask_tensor, replacement_image)
            deltas = _compute_batch_deltas(
                predictor, batch_inputs, original_logit, target_class_idx
            )
            all_deltas.extend(deltas)

        return all_deltas


def select_top_hyperpixels(
    hyperpixels: list[HyperpixelResult], max_hyperpixels: int = 10
) -> list[HyperpixelResult]:
    """Select top hyperpixels by their primary algorithm-specific score."""
    return sorted(
        hyperpixels,
        key=lambda hp: abs(hp.score),
        reverse=True,
    )[:max_hyperpixels]
