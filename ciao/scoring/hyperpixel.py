import torch

from ciao.model.predictor import ModelPredictor


def calculate_hyperpixel_deltas(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: torch.Tensor,
    hyperpixel_segment_ids_list: list[list[int]],
    replacement_image: torch.Tensor,
    target_class_idx: int,
    batch_size: int = 64,
) -> list[float]:
    """Calculate masking deltas for hyperpixel candidates using batched inference.

    Handles internal batching to prevent memory overflow with large path counts.

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch [1, C, H, W]
        segments: Pixel-to-segment mapping tensor [H, W]
        hyperpixel_segment_ids_list: List of segment ID lists, e.g. [[1,2,3], [4,5,6]]
        replacement_image: Replacement tensor [C, H, W]
        target_class_idx: Target class index
        batch_size: Batch size

    Returns:
        List[float]: Delta scores for each candidate
    """
    if not hyperpixel_segment_ids_list:
        return []

    # Validate all segment lists are non-empty
    for i, segment_ids in enumerate(hyperpixel_segment_ids_list):
        if not segment_ids:
            raise ValueError(f"Empty segment list at index {i}")

    if input_batch.dim() != 4 or input_batch.shape[0] != 1:
        raise ValueError(
            f"input_batch must have shape [1, C, H, W], got {tuple(input_batch.shape)}"
        )

    expected_shape = input_batch.shape[1:]
    if tuple(replacement_image.shape) != tuple(expected_shape):
        raise ValueError(
            "replacement_image must have shape [C, H, W] matching input_batch, "
            f"got {tuple(replacement_image.shape)} vs expected {tuple(expected_shape)}"
        )

    # Move tensors to the predictor's device once to avoid repeated transfers.
    # Align replacement_image dtype with input_batch to prevent torch.where errors.
    input_batch = input_batch.to(predictor.device)
    replacement_image = replacement_image.to(
        device=predictor.device, dtype=input_batch.dtype
    )

    with torch.no_grad():
        original_logit = predictor.get_class_logit_batch(input_batch, target_class_idx)[
            0
        ]

        gpu_segments = segments.to(predictor.device)

        all_deltas = []
        num_masks = len(hyperpixel_segment_ids_list)

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        for batch_start in range(0, num_masks, batch_size):
            batch_end = min(batch_start + batch_size, num_masks)
            current_batch_size = batch_end - batch_start

            # Clone on GPU directly
            batch_inputs = input_batch.repeat(current_batch_size, 1, 1, 1)

            # Fully vectorized mask creation
            mask_list = []
            for segment_ids in hyperpixel_segment_ids_list[batch_start:batch_end]:
                target_ids = torch.tensor(
                    segment_ids, dtype=gpu_segments.dtype, device=predictor.device
                )
                mask_list.append(torch.isin(gpu_segments, target_ids))

            # mask_tensor shape: [batch_size, H, W]
            mask_tensor = torch.stack(mask_list)

            # Apply masks using a single broadcasted operation
            batch_inputs = torch.where(
                mask_tensor.unsqueeze(1),  # [batch_size, 1, H, W]
                replacement_image.unsqueeze(0),  # [1, C, H, W]
                batch_inputs,  # [batch_size, C, H, W]
            )

            masked_logits = predictor.get_class_logit_batch(
                batch_inputs, target_class_idx
            )
            batch_deltas_tensor = original_logit - masked_logits
            all_deltas.extend(batch_deltas_tensor.tolist())

            del batch_inputs, masked_logits, mask_tensor

        return all_deltas


def select_top_hyperpixels(
    hyperpixels: list[dict[str, object]], max_hyperpixels: int = 10
) -> list[dict[str, object]]:
    """Select top hyperpixels by their primary algorithm-specific score."""
    return sorted(
        hyperpixels,
        key=lambda hp: abs(hp["hyperpixel_score"]),  # type: ignore[arg-type]
        reverse=True,
    )[:max_hyperpixels]
