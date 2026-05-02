"""Meaningful-Perturbations explainer producing a CIAO-compatible ExplanationResult."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ciao.baselines.meaningful_perturbations.mask_optimization import (
    MPResult,
    meaningful_perturbation,
)
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.explainer.ciao_explainer import ExplanationResult
from ciao.scoring.region import (
    RegionResult,
    calculate_region_deltas,
    calculate_region_probability_drops,
    log_odds_for_class,
)


if TYPE_CHECKING:
    from ciao.model.predictor import ModelPredictor
    from ciao.typing import ReplacementFn


# The mask identifies the region to perturb (delete); that region is the "important" one —
# the model breaks when it is removed.
_KEPT_SEGMENT_ID = 1
_BACKGROUND_SEGMENT_ID = 0


def _topk_pixel_mask(soft_mask: torch.Tensor, area: float) -> torch.Tensor:
    """Return a boolean mask selecting the top-`area` fraction of pixels by mask value."""
    flat = soft_mask.flatten()
    n_keep = max(1, round(area * flat.numel()))
    n_keep = min(n_keep, flat.numel())
    threshold = torch.topk(flat, n_keep).values.min()
    return soft_mask >= threshold


class MeaningfulPerturbationsExplainer:
    """Fong & Vedaldi 2017 meaningful-perturbations wrapped to produce an ExplanationResult.

    Pipeline:
        1. Optimize a deletion mask: find the smallest region whose removal
           (replacement with the supplied replacement image) kills the model's
           prediction for the target class.
        2. Threshold the soft mask to the top-`area` fraction → discrete region.
        3. Treat as 2-segment segmentation and score via log-odds drop.
    """

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        replacement: ReplacementFn,
        *,
        target_class_idx: int | None = None,
        area: float = 0.1,
        area_lambda: float = 8.0,
        tv_lambda: float = 1e-2,
        max_time: float = 60.0,
        max_iterations: int = 800,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        mask_step: int = 7,
        mask_sigma: float = 21.0,
        jitter: bool = True,
        batch_size: int = 64,
    ) -> tuple[ExplanationResult, MPResult]:
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")

        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)
        replacement_image = replacement(input_tensor)

        original_logits = predictor.get_logits(input_batch)
        original_probs = torch.nn.functional.softmax(original_logits, dim=1)
        if target_class_idx is None:
            target_class_idx = int(original_logits.argmax(dim=1)[0].item())

        original_prob = float(original_probs[0, target_class_idx].item())
        original_log_odds_tensor = log_odds_for_class(
            original_logits, target_class_idx
        )[0]
        original_log_odds = float(original_log_odds_tensor.item())

        mp_result = meaningful_perturbation(
            model=predictor.model,
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            replacement_image=replacement_image,
            area_lambda=area_lambda,
            tv_lambda=tv_lambda,
            max_time=max_time,
            max_iterations=max_iterations,
            learning_rate=learning_rate,
            momentum=momentum,
            mask_step=mask_step,
            mask_sigma=mask_sigma,
            jitter=jitter,
        )

        # Threshold to the top-`area` fraction of the deletion mask.
        kept_mask = _topk_pixel_mask(mp_result.soft_mask, area)
        segments = torch.where(
            kept_mask,
            torch.tensor(_KEPT_SEGMENT_ID, dtype=torch.int32, device=kept_mask.device),
            torch.tensor(
                _BACKGROUND_SEGMENT_ID, dtype=torch.int32, device=kept_mask.device
            ),
        )

        kept_region: frozenset[int] = frozenset({_KEPT_SEGMENT_ID})

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
            evaluations_count=mp_result.iterations,
            trajectory=list(mp_result.trajectory),
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

        segment_scores = {
            _BACKGROUND_SEGMENT_ID: 0.0,
            _KEPT_SEGMENT_ID: region_score,
        }
        class_names = predictor.class_names

        return ExplanationResult(
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            class_name=class_names[target_class_idx],
            original_log_odds=original_log_odds,
            segments=segments,
            segment_scores=segment_scores,
            regions=[region_result],
            replacement_image=replacement_image,
            soft_mask=mp_result.soft_mask,
        ), mp_result
