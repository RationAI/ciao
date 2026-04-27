"""CIAO explainer implementation."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ciao.algorithm.builder import build_all_regions
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.model.predictor import ModelPredictor
from ciao.scoring.region import RegionResult
from ciao.scoring.segments import (
    calculate_segment_scores,
    create_surrogate_dataset,
)
from ciao.typing import ExplanationMethodFn, ReplacementFn, SegmentationFn


@dataclass
class ExplanationResult:
    """Artifacts, regions and metadata required to produce visualizations."""

    input_batch: torch.Tensor
    target_class_idx: int
    class_name: str
    original_logit: float
    segments: torch.Tensor
    segment_scores: dict[int, float]  # Segment ID -> score
    regions: list[RegionResult]
    replacement_image: torch.Tensor


class CIAOExplainer:
    """CIAO (Contextual Importance Assessment via Obfuscation) Explainer.

    Generates explanations for image classification models by identifying
    influential image regions using mutual information and search algorithms.
    """

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        segmentation: SegmentationFn,
        method: ExplanationMethodFn,
        replacement: ReplacementFn,
        target_class_idx: int | None = None,
        max_regions: int = 10,
        desired_length: int = 30,
        batch_size: int = 64,
    ) -> ExplanationResult:
        """Generate CIAO explanation for an image.

        Args:
            image_path: Path to image file (string or pathlib.Path)
            predictor: ModelPredictor instance
            segmentation: Image segmentation function returning an ImageGraph.
            method: Explanation method function evaluating search contexts sequentially.
            replacement: Replacement strategy function generating an obfuscation mask.
            target_class_idx: Target class to explain (None = auto-select)
            max_regions: Maximum number of regions to build
            desired_length: Target number of segments per region (default=30)
            batch_size: Batch size for model evaluation

        Returns:
            ExplanationResult: ExplanationResult dataclass containing explanation artifacts and stats.
        """
        # 1. Boundary Validations
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        if not isinstance(predictor, ModelPredictor):
            raise TypeError(
                f"predictor must be a ModelPredictor instance, got {type(predictor).__name__}"
            )

        if max_regions <= 0:
            raise ValueError(f"max_regions must be positive, got {max_regions}")
        if desired_length <= 0:
            raise ValueError(f"desired_length must be positive, got {desired_length}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        class_names = predictor.class_names
        if target_class_idx is not None and (
            target_class_idx < 0 or target_class_idx >= len(class_names)
        ):
            raise ValueError(
                f"target_class_idx {target_class_idx} is out of bounds (0 to {len(class_names) - 1})"
            )

        # 2. Setup of the image
        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        if input_batch.dim() != 4 or input_batch.shape[0] != 1:
            raise ValueError(
                f"input_batch must have shape [1, C, H, W], got {tuple(input_batch.shape)}"
            )

        replacement_image = replacement(input_tensor)

        # Validate replacement output
        expected_shape = input_batch.shape[1:]
        if tuple(replacement_image.shape) != tuple(expected_shape):
            raise ValueError(
                "replacement_image must have shape [C, H, W] matching input_batch, "
                f"got {tuple(replacement_image.shape)} vs expected {tuple(expected_shape)}"
            )
        if replacement_image.device != input_tensor.device:
            raise ValueError(
                f"replacement_image device ({replacement_image.device}) must match "
                f"input_tensor device ({input_tensor.device})"
            )

        # 3. Compute base logits/probabilities once and resolve target class.
        original_logits = predictor.get_logits(input_batch)
        original_probs = torch.nn.functional.softmax(original_logits, dim=1)

        if target_class_idx is None:
            target_class_idx = int(original_logits.argmax(dim=1)[0].item())

            if target_class_idx < 0 or target_class_idx >= len(class_names):
                raise ValueError(
                    f"Model predicted class index {target_class_idx}, but class_names "
                    f"only has {len(class_names)} items. Check predictor configuration."
                )

        original_logit_tensor = original_logits[0, target_class_idx]
        original_logit = float(original_logit_tensor.item())
        original_prob = float(original_probs[0, target_class_idx].item())

        # 4. Create segmentation
        image_graph = segmentation(input_tensor)

        # 5. Calculate base scores from surrogate dataset
        X, y = create_surrogate_dataset(
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            original_logit=original_logit_tensor,
            batch_size=batch_size,
        )
        segment_scores = calculate_segment_scores(X, y)

        # 6. Execute the builder loop
        regions = build_all_regions(
            method=method,
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            scores=segment_scores,
            max_regions=max_regions,
            original_prob=original_prob,
            desired_length=desired_length,
            batch_size=batch_size,
        )

        class_name = class_names[target_class_idx]

        return ExplanationResult(
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            segments=image_graph.segments,
            segment_scores=segment_scores,
            regions=regions,
            class_name=class_name,
            replacement_image=replacement_image,
            original_logit=original_logit,
        )
