"""CIAO explainer implementation."""

from dataclasses import dataclass
from pathlib import Path

import torch

from ciao.algorithm.builder import build_all_hyperpixels
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.data.replacement import ReplacementFn
from ciao.data.segmentation import SegmentationFn
from ciao.explainer.explanation_methods import ExplanationMethodFn
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import HyperpixelResult
from ciao.scoring.segments import (
    calculate_segment_scores,
    create_surrogate_dataset,
)


@dataclass
class ExplanationResult:
    """Artifacts, hyperpixels and metadata required to produce visualizations."""

    input_batch: torch.Tensor
    target_class_idx: int
    class_name: str
    segments: torch.Tensor
    segment_scores: dict[int, float]  # Segment ID -> score
    hyperpixels: list[HyperpixelResult]


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
        max_hyperpixels: int = 10,
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
            max_hyperpixels: Maximum number of hyperpixels to build
            desired_length: Target number of segments per hyperpixel (default=30)
            batch_size: Batch size for model evaluation

        Returns:
            ExplanationResult: ExplanationResult dataclass containing explanation artifacts and stats.
        """
        # Get class names from predictor
        class_names = predictor.class_names

        # 1. Load and preprocess image (use same device as predictor's model)
        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        replacement_image = replacement(input_tensor)

        # 2. Get target class
        if target_class_idx is None:
            target_class_idx = predictor.get_predicted_class(input_batch)

        # 3. Create segmentation
        image_graph = segmentation(input_tensor)
        num_segments = image_graph.num_segments

        # Fail if segmentation is empty
        if num_segments == 0:
            raise ValueError(
                "Cannot generate explanation: The image contains 0 segments. "
                "Please check your segmentation algorithm and parameters."
            )

        # 4. Calculate base scores from surrogate dataset
        X, y = create_surrogate_dataset(
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
        )
        segment_scores = calculate_segment_scores(X, y)

        # 5. Execute the builder loop
        hyperpixels = build_all_hyperpixels(
            method=method,
            predictor=predictor,
            input_batch=input_batch,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            scores=segment_scores,
            max_hyperpixels=max_hyperpixels,
            desired_length=desired_length,
            batch_size=batch_size,
        )

        class_name = (
            class_names[target_class_idx]
            if target_class_idx < len(class_names)
            else f"Class {target_class_idx}"
        )

        # Return results
        return ExplanationResult(
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            segments=image_graph.segments,
            segment_scores=segment_scores,
            hyperpixels=hyperpixels,
            class_name=class_name,
        )
