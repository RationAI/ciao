"""Captum-LIME explainer producing the same `ExplanationResult` shape as CIAO."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from captum.attr import Lime

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
    from ciao.typing import ReplacementFn, SegmentationFn


SeedSelectionMode = Literal[1, -1] | None


def _per_segment_weights(
    attributions: torch.Tensor, segments: torch.Tensor
) -> dict[int, float]:
    """Reduce a [1, C, H, W] LIME attribution map to a per-segment mean.

    Captum-LIME assigns the same attribution value to every pixel inside a
    feature group, so any reduction (mean across channels + segment) recovers
    the per-segment LIME weight.
    """
    attr_2d = attributions.squeeze(0).mean(dim=0)  # [H, W]
    seg_ids = torch.unique(segments).tolist()
    weights: dict[int, float] = {}
    for seg_id in seg_ids:
        mask = segments == seg_id
        if mask.any():
            weights[int(seg_id)] = float(attr_2d[mask].mean().item())
    return weights


def _select_top_segments(
    weights: dict[int, float], desired_length: int, sigma: SeedSelectionMode
) -> list[int]:
    """Pick top-K segments respecting CIAO's sigma sign convention.

    sigma=None: rank by |weight| (the strongest evidence regardless of sign);
    sigma=1: keep only positive weights, descending;
    sigma=-1: keep only negative weights, ascending (most negative first).
    """
    items = list(weights.items())
    if sigma is None:
        items.sort(key=lambda kv: abs(kv[1]), reverse=True)
    elif sigma == 1:
        items = [kv for kv in items if kv[1] > 0]
        items.sort(key=lambda kv: kv[1], reverse=True)
    elif sigma == -1:
        items = [kv for kv in items if kv[1] < 0]
        items.sort(key=lambda kv: kv[1])
    else:
        raise ValueError(f"sigma must be None, 1, or -1, got {sigma!r}")

    return [seg_id for seg_id, _ in items[:desired_length]]


class LimeExplainer:
    """Captum-LIME wrapped to output a CIAO-compatible `ExplanationResult`.

    The user's `segmentation` is reused as Captum's `feature_mask`, the
    `replacement` is reused as the `baselines`, and the explanation region
    is the top-`desired_length` segments selected by sigma.
    """

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        segmentation: SegmentationFn,
        replacement: ReplacementFn,
        *,
        target_class_idx: int | None = None,
        sigma: SeedSelectionMode = None,
        desired_length: int = 30,
        n_samples: int = 1000,
        batch_size: int = 64,
    ) -> ExplanationResult:
        # Validation
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found at: {image_path}")
        if not isinstance(predictor, ModelPredictor):
            raise TypeError(
                f"predictor must be a ModelPredictor instance, got {type(predictor).__name__}"
            )
        if desired_length <= 0:
            raise ValueError(f"desired_length must be positive, got {desired_length}")
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        class_names = predictor.class_names
        if target_class_idx is not None and (
            target_class_idx < 0 or target_class_idx >= len(class_names)
        ):
            raise ValueError(
                f"target_class_idx {target_class_idx} is out of bounds (0 to {len(class_names) - 1})"
            )

        # Load and preprocess
        input_tensor = load_and_preprocess_image(image_path, device=predictor.device)
        input_batch = input_tensor.unsqueeze(0)
        replacement_image = replacement(input_tensor)

        # Resolve target class
        original_logits = predictor.get_logits(input_batch)
        original_probs = torch.nn.functional.softmax(original_logits, dim=1)
        if target_class_idx is None:
            target_class_idx = int(original_logits.argmax(dim=1)[0].item())

        original_prob = float(original_probs[0, target_class_idx].item())
        original_log_odds_tensor = log_odds_for_class(
            original_logits, target_class_idx
        )[0]
        original_log_odds = float(original_log_odds_tensor.item())

        # Segmentation provides feature groups for LIME
        image_graph = segmentation(input_tensor)
        segments = image_graph.segments  # [H, W], int

        # Captum wants a feature_mask broadcastable to inputs ([1, C, H, W]).
        # Same group ID across channels -> attribution shared across channels.
        feature_mask = segments.long().unsqueeze(0).unsqueeze(0).to(predictor.device)
        baselines = replacement_image.unsqueeze(0).to(predictor.device)

        lime = Lime(predictor.model)
        attributions = lime.attribute(
            input_batch,
            baselines=baselines,
            target=target_class_idx,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=batch_size,
        )

        segment_weights = _per_segment_weights(attributions, segments)
        top_segment_ids = _select_top_segments(
            segment_weights, desired_length=desired_length, sigma=sigma
        )

        kept_region = frozenset(top_segment_ids)

        if kept_region:
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
        else:
            region_score = 0.0

        region_result = RegionResult(
            region=kept_region,
            score=region_score,
            evaluations_count=n_samples,
            trajectory=[],
        )

        if kept_region:
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
        else:
            region_result.original_prob = original_prob
            region_result.masked_prob = original_prob
            region_result.probability_drop = 0.0
            region_result.masked_top_class_idx = target_class_idx
            region_result.masked_top_class_name = class_names[target_class_idx]
            region_result.masked_top_prob = original_prob

        return ExplanationResult(
            input_batch=input_batch,
            target_class_idx=target_class_idx,
            class_name=class_names[target_class_idx],
            original_log_odds=original_log_odds,
            segments=segments,
            segment_scores=segment_weights,
            regions=[region_result] if kept_region else [],
            replacement_image=replacement_image,
        )
