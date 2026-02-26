"""CIAO explainer implementation."""

import logging
from pathlib import Path
from typing import Any

import torch

from ciao.algorithm.lookahead_bitset import build_all_hyperpixels_greedy_lookahead
from ciao.algorithm.mcgs import build_all_hyperpixels_mcgs
from ciao.algorithm.mcts import build_all_hyperpixels_mcts
from ciao.algorithm.potential import build_all_hyperpixels_potential
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.utils.calculations import (
    ModelPredictor,
    calculate_scores_from_surrogate,
    create_surrogate_dataset,
    get_predicted_class,
    select_top_hyperpixels,
)
from ciao.utils.segmentation import (
    build_adjacency_bitmasks,
    create_segmentation,
    graph_to_adjacency_list,
)


logger = logging.getLogger(__name__)


class CIAOExplainer:
    """CIAO (Contextual Importance Assessment via Obfuscation) Explainer.

    Generates explanations for image classification models by identifying
    influential image regions using mutual information and greedy search.
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
        segmentation_type: str = "hexagonal",
        max_hyperpixels: int = 10,
        desired_length: int = 30,
        batch_size: int = 64,
        neighborhood: int = 8,
        replacement: str = "mean_color",
        replacement_kwargs: dict[str, Any] | None = None,
        method_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate CIAO explanation for an image.

        Args:
            image_path: Path to image or PIL Image object
            predictor: ModelPredictor instance
            method: Hyperpixel construction method. Options:
                - "potential": Potential field guided search
                - "mcts": Monte Carlo Tree Search
                - "mc_rave": MC-RAVE (MCTS with RAVE heuristic)
                - "lookahead": Optimized greedy lookahead with bitsets (default)
                - "mcgs": Monte Carlo Graph Search
                - "mcgs_rave": MCGS with RAVE
            target_class_idx: Target class to explain (None = auto-select)
            segment_size: Size of segments in pixels
            segmentation_type: Type of segmentation ("hexagonal")
            max_hyperpixels: Maximum number of hyperpixels to build
            desired_length: Target number of segments per hyperpixel (default=30)
            batch_size: Batch size for model evaluation
            neighborhood: Adjacency neighborhood (6 or 8 for hexagonal)
            replacement: Masking strategy for model evaluation
            replacement_kwargs: Additional kwargs for replacement method
            method_params: Dictionary of method-specific parameters:

                For "potential":
                    - num_simulations: int (default=50) - Number of simulations

                For "mcts":
                    - num_iterations: int (default=100) - MCTS iterations
                    - exploration_c: float (default=1.4) - UCT exploration constant
                    - mcts_batch_size: int (default=64) - Batch size for MCTS

                For "mc_rave":
                    - num_iterations: int (default=100)
                    - exploration_c: float (default=1.4)
                    - mcts_batch_size: int (default=64)
                    - rave_k: float (default=1000)

                For "lookahead":
                    - lookahead_distance: int (default=2)

                For "mcgs":
                    - num_iterations: int (default=100)
                    - mcts_batch_size: int (default=64)
                    - exploration_c: float (default=1.4)

                For "mcgs_rave":
                    - num_iterations: int (default=100)
                    - mcts_batch_size: int (default=64)
                    - exploration_c: float (default=1.4)
                    - rave_k: float (default=1000)

        Returns:
            Dictionary containing:
                - input_batch: Preprocessed input tensor
                - target_class_idx: Class being explained
                - segments: Segmentation map
                - scores: Individual segment scores
                - hyperpixels: List of all hyperpixels found
                - top_hyperpixels: Top-k hyperpixels by score
                - class_name: Human-readable class name
                - performance_mode: Method identifier
        """
        # Input validation
        valid_methods = [
            "potential",
            "mcts",
            "mc_rave",
            "lookahead",
            "mcgs",
            "mcgs_rave",
        ]
        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}. Valid options are: {', '.join(valid_methods)}"
            )

        if max_hyperpixels <= 0:
            raise ValueError(f"max_hyperpixels must be positive, got {max_hyperpixels}")

        if desired_length <= 0:
            raise ValueError(f"desired_length must be positive, got {desired_length}")

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        if segment_size <= 0:
            raise ValueError(f"segment_size must be positive, got {segment_size}")

        if segmentation_type not in ["hexagonal", "square"]:
            raise ValueError(
                f"Invalid segmentation_type: {segmentation_type}. "
                "Valid options are: 'hexagonal', 'square'"
            )

        # Initialize method params with defaults
        if method_params is None:
            method_params = {}

        # Get class names from predictor
        class_names = predictor.class_names

        # 1. Load and preprocess image (use same device as predictor's model)
        input_batch, _original_image, input_tensor = load_and_preprocess_image(
            image_path, device=predictor.device
        )

        # Handle replacement kwargs
        if replacement_kwargs is None:
            replacement_kwargs = {}

        predictor.replacement_image = predictor.get_replacement_image(
            input_tensor, replacement, **replacement_kwargs
        ).to(predictor.device)

        # 2. Get target class
        if target_class_idx is None:
            target_class_idx = get_predicted_class(predictor, input_batch)
            logger.info(f"Auto-selected target class: {target_class_idx}")
        else:
            # Validate target_class_idx if provided
            num_classes = len(class_names) if class_names else None
            if num_classes and (
                target_class_idx >= num_classes or target_class_idx < 0
            ):
                raise ValueError(
                    f"target_class_idx {target_class_idx} is out of range. "
                    f"Model has {num_classes} classes (indices 0-{num_classes - 1})"
                )

        # 3. Create segmentation
        segments, graph = create_segmentation(
            input_tensor,
            segmentation_type=segmentation_type,
            segment_size=segment_size,
            neighborhood=neighborhood,
        )
        logger.info(
            f"Built {segmentation_type} spatial graph with {graph.number_of_nodes()} "
            f"segments and {graph.number_of_edges()} edges"
        )

        # Calculate scores from surrogate dataset
        X, y = create_surrogate_dataset(
            predictor,
            input_batch,
            segments,
            graph,
            target_class_idx,
            batch_size=batch_size,
        )
        scores = calculate_scores_from_surrogate(X, y)

        # Create adjacency structures (needed by all methods)
        # Use the same segmentation for consistency between scoring and search
        num_segments = graph.number_of_nodes()
        adj_list = graph_to_adjacency_list(graph, num_segments)
        adj_masks = build_adjacency_bitmasks(adj_list)

        # Build hyperpixels based on method
        if method == "potential":
            hyperpixels = build_all_hyperpixels_potential(
                predictor=predictor,
                input_batch=input_batch,
                segments=segments,
                adj_masks=adj_masks,
                target_class_idx=target_class_idx,
                scores=scores,
                max_hyperpixels=max_hyperpixels,
                desired_length=desired_length,
                num_simulations=method_params.get("num_simulations", 50),
                batch_size=batch_size,
            )

        elif method in ["mcts", "mc_rave"]:
            mode_str = "rave" if method == "mc_rave" else "standard"

            hyperpixels = build_all_hyperpixels_mcts(
                predictor=predictor,
                input_batch=input_batch,
                segments=segments,
                adj_masks=adj_masks,
                target_class_idx=target_class_idx,
                scores=scores,
                max_hyperpixels=max_hyperpixels,
                desired_length=desired_length,
                num_iterations=method_params.get("num_iterations", 100),
                mode=mode_str,
                batch_size=method_params.get("mcts_batch_size", 64),
                exploration_c=method_params.get("exploration_c", 1.4),
                rave_k=method_params.get("rave_k", 1000),
            )

        elif method == "lookahead":
            hyperpixels = build_all_hyperpixels_greedy_lookahead(
                predictor=predictor,
                input_batch=input_batch,
                segments=segments,
                adj_masks=adj_masks,
                target_class_idx=target_class_idx,
                scores=scores,
                max_hyperpixels=max_hyperpixels,
                desired_length=desired_length,
                lookahead_distance=method_params.get("lookahead_distance", 2),
                batch_size=batch_size,
            )

        elif method in ["mcgs", "mcgs_rave"]:
            # Determine mode based on method
            mode_str = "rave" if method == "mcgs_rave" else "standard"

            hyperpixels = build_all_hyperpixels_mcgs(
                predictor=predictor,
                input_batch=input_batch,
                segments=segments,
                adj_masks=adj_masks,
                target_class_idx=target_class_idx,
                scores=scores,
                max_hyperpixels=max_hyperpixels,
                desired_length=desired_length,
                num_iterations=method_params.get("num_iterations", 100),
                mode=mode_str,
                batch_size=method_params.get("mcts_batch_size", 64),
                exploration_c=method_params.get("exploration_c", 1.4),
                rave_k=method_params.get("rave_k", 1000.0),
            )

        else:
            raise ValueError(
                f"Unknown method: {method}. Valid options: potential, mcts, "
                f"mc_rave, lookahead, mcgs, mcgs_rave"
            )

        # Select top hyperpixels
        top_hyperpixels = select_top_hyperpixels(hyperpixels, max_hyperpixels)

        logger.info(f"Class name: {class_names[target_class_idx]}")

        # Return results
        result = {
            "input_batch": input_batch,
            "target_class_idx": target_class_idx,
            "segments": segments,
            "scores": scores,
            "hyperpixels": hyperpixels,
            "top_hyperpixels": top_hyperpixels,
            "class_name": class_names[target_class_idx]
            if target_class_idx < len(class_names)
            else f"Class {target_class_idx}",
            "performance_mode": method,
        }
        return result

    def visualize(
        self,
        image: torch.Tensor,
        explanation: dict[str, Any],
        save_path: str | Path | None = None,
        interactive: bool = True,
    ) -> Any:
        """Visualize explanation results.

        Args:
            image: Input image tensor
            explanation: Explanation dictionary from explain()
            save_path: Optional path to save visualization
            interactive: Whether to display interactive visualization

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError(
            "The visualize method is not yet implemented. "
            "Use external visualization tools or implement custom visualization based on the explanation data."
        )
