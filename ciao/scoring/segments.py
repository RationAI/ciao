import logging

import numpy as np
import numpy.typing as npt
import torch

from ciao.algorithm.graph import ImageGraph
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import calculate_hyperpixel_deltas


logger = logging.getLogger(__name__)


def create_surrogate_dataset(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    replacement_image: torch.Tensor,
    image_graph: ImageGraph,
    target_class_idx: int,
    neighborhood_distance: int = 1,
    batch_size: int = 16,
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.float32]]:
    """Create surrogate dataset for interpretability.

    Each row represents one masking operation:
    - Features (X): Binary indicator vector [num_segments] - 1 if segment was masked, 0 otherwise
    - Target (y): Delta score (original_logit - masked_logit)

    This dataset can be used for:
    - Computing segment importance scores
    - Training interpretable models (like LIME does)
    - Analyzing masking effects

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch
        replacement_image: Replacement tensor [C, H, W]
        image_graph: ImageGraph instance with segments and adj_list
        target_class_idx: Target class index
        neighborhood_distance: Distance for neighborhood masking
        batch_size: Batch size for processing segments

    Returns:
        X: Binary indicator matrix [num_samples, num_segments]
        y: Delta scores array [num_samples]
    """
    if neighborhood_distance < 0:
        raise ValueError(
            f"neighborhood_distance cannot be negative. Got {neighborhood_distance}."
        )
    if not image_graph.adj_list:
        raise ValueError("adj_list in image_graph cannot be empty.")

    # BFS algorithm
    local_groups = []
    num_segments = image_graph.num_segments

    for segment_id in range(num_segments):
        visited: set[int] = {segment_id}
        current_layer: frozenset[int] = frozenset({segment_id})

        for _ in range(neighborhood_distance):
            next_layer = image_graph.get_frontier(current_layer, visited)

            # Early exit if we reached the boundary of the isolated graph component
            if not next_layer:
                break

            visited |= next_layer
            current_layer = next_layer

        local_groups.append(sorted(visited))

    # Calculate deltas for all local groups
    deltas = calculate_hyperpixel_deltas(
        predictor,
        input_batch,
        image_graph.segments,
        local_groups,
        replacement_image,
        target_class_idx,
        batch_size=batch_size,
    )

    # Create surrogate dataset
    num_samples = len(local_groups)
    X = np.zeros((num_samples, num_segments), dtype=np.int8)
    y = np.array(deltas, dtype=np.float32)

    # Fast vectorized indicator matrix filling
    for i, masked_segments in enumerate(local_groups):
        X[i, masked_segments] = 1

    logger.info(f"Created surrogate dataset: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Average delta: {y.mean():.4f}, std: {y.std():.4f}")

    return X, y


def calculate_segment_scores(
    X: npt.NDArray[np.int8],  # noqa: N803
    y: npt.NDArray[np.float32],
) -> dict[int, float]:
    """Calculate neighborhood-smoothed segment importance scores from sampled deltas.

    This function computes the mean delta score for each segment across all
    surrogate samples where that segment was masked. It acts as a fast
    approximation of the segment's marginal contribution to the prediction.

    Args:
        X: Binary indicator matrix of shape [num_samples, num_segments].
           X[i, j] == 1 if segment j is masked in sample i, else 0.
        y: Delta scores array of shape [num_samples].

    Returns:
        Dict mapping segment_id -> averaged importance score.

    Raises:
        ValueError: If any segment was never masked in the surrogate dataset.
    """
    # Vectorized count of how many times each segment was masked
    counts = X.sum(axis=0)

    # Fail-fast validation for unmasked segments
    unmasked_indices = np.asarray(counts == 0).nonzero()[0]
    if unmasked_indices.size > 0:
        raise ValueError(
            f"Segment(s) {unmasked_indices.tolist()} never appear in any local group. "
            "This suggests a bug in group generation or segment ID mapping."
        )

    # Vectorized mean calculation using matrix multiplication
    segment_means = (y @ X) / counts

    # Convert the numpy array results to the expected dictionary format
    scores = {int(i): float(score) for i, score in enumerate(segment_means)}

    if scores:
        score_values = list(scores.values())
        logger.info(f"Score range: [{min(score_values):.4f}, {max(score_values):.4f}]")

    return scores
