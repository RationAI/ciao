from collections.abc import Sequence, Set
from dataclasses import dataclass, field

import torch

from ciao.model.predictor import ModelPredictor


@dataclass
class RegionResult:
    """Type definition for the output of region building algorithms."""

    region: frozenset[int]
    score: float

    # Tracking metrics (populated after search completes)
    original_prob: float = 0.0
    masked_prob: float = 0.0
    probability_drop: float = 0.0
    evaluations_count: int = 0
    trajectory: list[dict[str, float]] = field(default_factory=list)

    # Top class on the input with this region masked out (populated after search completes)
    masked_top_class_idx: int = 0
    masked_top_class_name: str = ""
    masked_top_prob: float = 0.0


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


def calculate_region_deltas(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    segment_sets: Sequence[Set[int]],
    replacement_image: torch.Tensor,
    target_class_idx: int,
    original_logit: torch.Tensor,
    batch_size: int = 64,
) -> list[float]:
    """Calculate masking deltas for region candidates using batched inference.

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch [1, C, H, W]
        segments: Pixel-to-segment mapping array [H, W]
        segment_sets: List of segment ID sets, e.g. [{1,2,3}, {4,5,6}]
        replacement_image: Replacement tensor [C, H, W]
        target_class_idx: Target class index
        original_logit: Pre-computed unmasked target-class logit
        batch_size: Batch size for internal batching

    Returns:
        Delta scores for each candidate
    """
    if not segment_sets:
        return []

    input_batch, replacement_image, gpu_segments = _prepare_tensors_for_model(
        predictor, input_batch, replacement_image, segments
    )

    with torch.no_grad():
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


def calculate_region_probability_drops(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    replacement_image: torch.Tensor,
    target_class_idx: int,
    original_prob: float,
    results: list[RegionResult],
    batch_size: int = 64,
) -> list[RegionResult]:
    """Compute masked class probabilities for all finished regions in batched passes.

    Populates ``original_prob``, ``masked_prob``, and ``probability_drop``
    on each given *result* (mutates in place **and** returns the list).

    Args:
        predictor: ModelPredictor instance
        input_batch: Original image tensor [1, C, H, W]
        segments: Pixel-to-segment mapping [H, W]
        replacement_image: Replacement tensor [C, H, W]
        target_class_idx: Target class index
        original_prob: Pre-computed unmasked probability for the target class
        results: RegionResults whose regions will be masked
        batch_size: Max regions per forward pass

    Returns:
        The same RegionResults with probability fields populated.
    """
    if not results:
        return results

    input_batch, replacement_image, gpu_segments = _prepare_tensors_for_model(
        predictor, input_batch, replacement_image, segments
    )

    region_sets = [r.region for r in results]
    masked_probs: list[float] = []
    top_class_idxs: list[int] = []
    top_class_probs: list[float] = []

    for batch_start in range(0, len(region_sets), batch_size):
        batch_end = min(batch_start + batch_size, len(region_sets))
        slice_ = region_sets[batch_start:batch_end]
        mask_tensor = _build_mask_tensor(gpu_segments, slice_, predictor.device)
        masked_input = _apply_masks(input_batch, mask_tensor, replacement_image)
        probs = predictor.get_predictions(masked_input)
        masked_probs.extend(probs[:, target_class_idx].tolist())
        top_probs, top_idxs = probs.max(dim=1)
        top_class_probs.extend(top_probs.tolist())
        top_class_idxs.extend(top_idxs.tolist())

    class_names = predictor.class_names
    for result, masked_prob, top_idx, top_prob in zip(
        results, masked_probs, top_class_idxs, top_class_probs, strict=True
    ):
        result.original_prob = original_prob
        result.masked_prob = masked_prob
        result.probability_drop = original_prob - masked_prob
        result.masked_top_class_idx = int(top_idx)
        result.masked_top_class_name = class_names[int(top_idx)]
        result.masked_top_prob = float(top_prob)

    return results


def select_top_regions(
    regions: list[RegionResult], max_regions: int = 10
) -> list[RegionResult]:
    """Select top regions by their primary algorithm-specific score."""
    return sorted(
        regions,
        key=lambda r: abs(r.score),
        reverse=True,
    )[:max_regions]
