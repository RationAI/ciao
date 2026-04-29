"""Extremal-Perturbations explainer: pixel-level kept region + prob-drop scoring."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ciao.baselines.extremal.perturbation import EPResult, extremal_perturbation
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.explainer.ciao_explainer import ExplanationResult
from ciao.model.predictor import ModelPredictor
from ciao.scoring.region import (
    RegionResult,
    calculate_region_deltas,
    calculate_region_probability_drops,
    log_odds_for_class,
)


if TYPE_CHECKING:
    from ciao.typing import ReplacementFn


# Segment ID assigned to pixels EP decided to keep (i.e. the "important" region).
# Used by the visualization layer, which expects a segments tensor + region IDs.
_KEPT_SEGMENT_ID = 1
_BACKGROUND_SEGMENT_ID = 0


def _topk_pixel_mask(soft_mask: torch.Tensor, area: float) -> torch.Tensor:
    """Return a boolean mask selecting the top-`area` fraction of pixels.

    Using top-K instead of a 0.5 threshold guarantees the discrete kept
    region has exactly the requested area regardless of how well the
    optimizer hit the constraint.
    """
    flat = soft_mask.flatten()
    n_keep = max(1, round(area * flat.numel()))
    n_keep = min(n_keep, flat.numel())
    threshold = torch.topk(flat, n_keep).values.min()
    return soft_mask >= threshold


class ExtremalPerturbationExplainer:
    """Wraps EP optimization into the same `ExplanationResult` contract as CIAO.

    Pipeline:
        1. Run EP preservation-game optimization to find a smooth mask of
           fixed area that maximizes the target class probability.
        2. Threshold the soft mask to get a discrete "kept" pixel region.
        3. Treat that as a 2-segment segmentation (kept vs. background) so
           the existing scoring + visualization helpers can be reused.
        4. Replace the kept region with the replacement image, run a single
           inference, and report log-odds drop / probability drop.
    """

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        replacement: ReplacementFn,
        *,
        target_class_idx: int | None = None,
        area: float = 0.1,
        max_time: float = 60.0,
        max_iterations: int = 800,
        learning_rate: float = 0.05,
        mask_step: int = 7,
        mask_sigma: float = 21.0,
        area_lambda: float = 300.0,
        area_lambda_growth: float = 1.0035,
        batch_size: int = 64,
    ) -> tuple[ExplanationResult, EPResult]:
        # Validation
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        if not isinstance(predictor, ModelPredictor):
            raise TypeError(
                f"predictor must be a ModelPredictor instance, got {type(predictor).__name__}"
            )

        class_names = predictor.class_names
        if target_class_idx is not None and (
            target_class_idx < 0 or target_class_idx >= len(class_names)
        ):
            raise ValueError(
                f"target_class_idx {target_class_idx} is out of bounds (0 to {len(class_names) - 1})"
            )

        # Load image and replacement
        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)
        replacement_image = replacement(input_tensor)

        expected_shape = input_batch.shape[1:]
        if tuple(replacement_image.shape) != tuple(expected_shape):
            raise ValueError(
                "replacement_image must have shape [C, H, W] matching input_batch, "
                f"got {tuple(replacement_image.shape)} vs expected {tuple(expected_shape)}"
            )

        # Resolve target class against unmasked logits.
        original_logits = predictor.get_logits(input_batch)
        original_probs = torch.nn.functional.softmax(original_logits, dim=1)
        if target_class_idx is None:
            target_class_idx = int(original_logits.argmax(dim=1)[0].item())

        original_prob = float(original_probs[0, target_class_idx].item())
        original_log_odds_tensor = log_odds_for_class(
            original_logits, target_class_idx
        )[0]
        original_log_odds = float(original_log_odds_tensor.item())

        # Run EP
        ep_result = extremal_perturbation(
            model=predictor.model,
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            replacement_image=replacement_image,
            area=area,
            max_time=max_time,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            mask_step=mask_step,
            mask_sigma=mask_sigma,
            area_lambda=area_lambda,
            area_lambda_growth=area_lambda_growth,
        )

        # Discretize: top-area fraction of pixels = kept region.
        kept_mask = _topk_pixel_mask(ep_result.soft_mask, area)
        segments = torch.where(
            kept_mask,
            torch.tensor(_KEPT_SEGMENT_ID, dtype=torch.int32, device=kept_mask.device),
            torch.tensor(
                _BACKGROUND_SEGMENT_ID, dtype=torch.int32, device=kept_mask.device
            ),
        )

        kept_region: frozenset[int] = frozenset({_KEPT_SEGMENT_ID})

        # Score the discrete region: log-odds drop after replacing kept pixels.
        deltas = calculate_region_deltas(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            segment_sets=[kept_region],
            replacement_image=replacement_image,
            target_class_idx=target_class_idx,
            original_log_odds=original_log_odds_tensor,
            batch_size=batch_size,
        )
        region_score = float(deltas[0]) if deltas else 0.0

        region_result = RegionResult(
            region=kept_region,
            score=region_score,
            evaluations_count=ep_result.iterations,
            trajectory=list(ep_result.trajectory),
        )

        calculate_region_probability_drops(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            replacement_image=replacement_image,
            target_class_idx=target_class_idx,
            original_prob=original_prob,
            results=[region_result],
            batch_size=batch_size,
        )

        # 2-segment "score map" for the visualization heatmap pane.
        segment_scores = {
            _BACKGROUND_SEGMENT_ID: 0.0,
            _KEPT_SEGMENT_ID: region_score,
        }

        explanation = ExplanationResult(
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            class_name=class_names[target_class_idx],
            original_log_odds=original_log_odds,
            segments=segments,
            segment_scores=segment_scores,
            regions=[region_result],
            replacement_image=replacement_image,
        )
        return explanation, ep_result
