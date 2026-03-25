"""CIAO explainer implementation."""

import logging
from pathlib import Path
from typing import Any, Literal, TypedDict

import torch

from ciao.algorithm.builder import build_all_hyperpixels
from ciao.algorithm.lookahead import build_hyperpixel_greedy_lookahead
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.data.replacement import get_replacement_image
from ciao.data.segmentation import create_segmentation
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import HyperpixelResult
from ciao.scoring.segments import (
    calculate_segment_scores,
    create_surrogate_dataset,
)


logger = logging.getLogger(__name__)


class ExplanationResult(TypedDict):
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

    def __init__(self) -> None:
        """Initialize the CIAO explainer."""

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        method: str = "lookahead",
        target_class_idx: int | None = None,
        segment_size: int = 4,
        segmentation_type: Literal["square", "hexagonal"] = "hexagonal",
        max_hyperpixels: int = 10,
        desired_length: int = 30,
        batch_size: int = 64,
        neighborhood: int = 8,
        replacement: Literal[
            "mean_color", "interlacing", "blur", "solid_color"
        ] = "mean_color",
        replacement_kwargs: dict[str, Any] | None = None,
        method_params: dict[str, Any] | None = None,
    ) -> ExplanationResult:
        """Generate CIAO explanation for an image.

        Args:
            image_path: Path to image or PIL Image object
            predictor: ModelPredictor instance
            method: Hyperpixel construction method. Options:
                - "lookahead": Optimized greedy lookahead with bitsets (default)
                (Note: "mcts" and "mcgs" methods will be added in future updates)
            target_class_idx: Target class to explain (None = auto-select)
            segment_size: Size of segments in pixels
            segmentation_type: Type of segmentation ("hexagonal")
            max_hyperpixels: Maximum number of hyperpixels to build
            desired_length: Target number of segments per hyperpixel (default=30)
            batch_size: Batch size for model evaluation
            neighborhood: Adjacency neighborhood (6 or 8 for hexagonal)
            replacement: Masking strategy for model evaluation
            replacement_kwargs: Additional kwargs for replacement method
            method_params: Dictionary of method-specific parameters.
                For "lookahead":
                    - lookahead_distance: int (default=2)

        Returns:
            Dictionary containing explanation artifacts and stats.
        """
        # 0. Early validation of method to fail fast
        if method != "lookahead":
            raise NotImplementedError(
                f"Method '{method}' is not yet implemented. Currently, only "
                f"'lookahead' is supported."
            )

        # Initialize method params with defaults
        if method_params is None:
            method_params = {}
        if replacement_kwargs is None:
            replacement_kwargs = {}

        # Get class names from predictor
        class_names = predictor.class_names

        # 1. Load and preprocess image (use same device as predictor's model)
        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

        replacement_image = get_replacement_image(
            input_tensor, replacement, **replacement_kwargs
        ).to(predictor.device)

        # 2. Get target class
        if target_class_idx is None:
            target_class_idx = predictor.get_predicted_class(input_batch)
            logger.info(f"Auto-selected target class: {target_class_idx}")

        # 3. Create segmentation
        image_graph = create_segmentation(
            input_tensor,
            segmentation_type=segmentation_type,
            segment_size=segment_size,
            neighborhood=neighborhood,
        )
        num_segments = image_graph.num_segments

        # Fail if segmentation is empty
        if num_segments == 0:
            raise ValueError(
                "Cannot generate explanation: The image contains 0 segments. "
                "Please check your segmentation algorithm and parameters."
            )

        logger.info(
            f"Built {segmentation_type} segmentation with {num_segments} segments"
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

        # 5. Set up algorithm-specific builder and parameters
        algo_kwargs = {"desired_length": desired_length, "batch_size": batch_size}

        if method == "lookahead":
            builder_func = build_hyperpixel_greedy_lookahead
            algo_kwargs["lookahead_distance"] = method_params.get(
                "lookahead_distance", 2
            )

        # 6. Execute the builder loop
        hyperpixels = build_all_hyperpixels(
            builder_func=builder_func,
            predictor=predictor,
            input_batch=input_batch,
            segments=image_graph.segments,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            scores=segment_scores,
            max_hyperpixels=max_hyperpixels,
            **algo_kwargs,
        )

        class_name = (
            class_names[target_class_idx]
            if target_class_idx < len(class_names)
            else f"Class {target_class_idx}"
        )
        logger.info(f"Explanation built for class: {class_name}")

        # Return results
        result: ExplanationResult = {
            "input_batch": input_batch,
            "target_class_idx": target_class_idx,
            "segments": image_graph.segments,
            "segment_scores": segment_scores,
            "hyperpixels": hyperpixels,
            "class_name": class_name,
        }
        return result

    def visualize(
        self,
        image: torch.Tensor,
        explanation: ExplanationResult,
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> Any:
        """Visualize the generated explanation.

        Raises:
            NotImplementedError: Will be implemented soon.
        """
        raise NotImplementedError(
            "Visualization functionality will be (hopefully) added in a future update."
        )
