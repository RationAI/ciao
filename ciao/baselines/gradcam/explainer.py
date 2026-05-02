"""GradCAM / GradCAM++ explainer producing a CIAO-compatible ExplanationResult."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch

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


_KEPT_SEGMENT_ID = 1
_BACKGROUND_SEGMENT_ID = 0

GradCAMVariant = Literal["gradcam", "gradcampp"]


def _resolve_layer(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    """Resolve a dot-separated layer name like 'layer4' or 'layer4.1' on a model."""
    module: torch.nn.Module = model
    for part in layer_name.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)  # type: ignore[index]
    return module


def _topk_pixel_mask(heatmap: torch.Tensor, area: float) -> torch.Tensor:
    """Return a boolean mask selecting the top-`area` fraction of pixels by heatmap value."""
    flat = heatmap.flatten()
    n_keep = max(1, round(area * flat.numel()))
    n_keep = min(n_keep, flat.numel())
    threshold = torch.topk(flat, n_keep).values.min()
    return heatmap >= threshold


class GradCAMExplainer:
    """GradCAM or GradCAM++ wrapped to output a CIAO-compatible ExplanationResult.

    Pipeline:
        1. Run GradCAM/GradCAM++ to get a [H, W] heatmap.
        2. Threshold to top-`area` fraction → discrete kept region.
        3. Treat as 2-segment segmentation (kept vs background).
        4. Score via log-odds drop when the kept region is replaced.
    """

    def explain(
        self,
        image_path: str | Path,
        predictor: ModelPredictor,
        replacement: ReplacementFn,
        *,
        target_class_idx: int | None = None,
        variant: GradCAMVariant = "gradcam",
        target_layer: str = "layer4",
        area: float = 0.1,
        batch_size: int = 64,
    ) -> ExplanationResult:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

        layer_module = _resolve_layer(predictor.model, target_layer)
        cam_cls = GradCAMPlusPlus if variant == "gradcampp" else GradCAM
        cam = cam_cls(model=predictor.model, target_layers=[layer_module])

        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(input_tensor=input_batch, targets=targets)
        # grayscale_cam: [1, H, W], float32 in [0, 1]
        heatmap = torch.from_numpy(grayscale_cam[0]).to(predictor.device)

        kept_mask = _topk_pixel_mask(heatmap, area)
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
            evaluations_count=1,
            trajectory=[],
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
            soft_mask=heatmap,
        )
