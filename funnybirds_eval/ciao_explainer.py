"""CIAO explainer adapter for the FunnyBirds evaluation framework.

Converts CIAO's region search output into a pixel-level attribution map that
FunnyBirds' threshold-based metrics can consume.  Each pixel in a found region
receives the region's log-odds drop score as its attribution value; pixels
outside all regions receive 0.  This ensures that the metric results reflect
the *search algorithm's* decisions, not just the underlying segment scores.

Usage:
    explainer = CIAOFunnyBirdsExplainer(
        model=model,
        class_names=[str(i) for i in range(50)],
        method='lookahead',
        segmentation='hex',
        hex_radius=8,
        max_regions=5,
        desired_length=30,
    )
    # Then pass to evaluation_protocols as usual.

Requirements:
    The `ciao` package must be installed in the active Python environment.
    Install from the repo root:  pip install -e /path/to/ciao
"""

from __future__ import annotations

import time
from typing import ClassVar

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from explainers.explainer_wrapper import AbstractAttributionExplainer

from ciao.algorithm.builder import build_all_regions
from ciao.data.constants import IMAGENET_MEAN, IMAGENET_STD
from ciao.data.replacement import make_solid_color_replacement
from ciao.data.segmentation import make_hexagonal_segmentation, make_square_segmentation
from ciao.explainer.explanation_methods import (
    make_beam_search_method,
    make_lookahead_method,
    make_mcgs_method,
    make_mcts_method,
    make_potential_method,
    make_pure_monte_carlo_method,
    make_ucb_method,
)
from ciao.model.predictor import ModelPredictor
from ciao.scoring.region import log_odds_for_class
from ciao.scoring.segments import calculate_segment_scores, create_surrogate_dataset
from ciao.typing import ExplanationMethodFn

matplotlib.use("Agg")

_replacement_fn = make_solid_color_replacement(color=(85, 85, 153))


def _make_explanation_figure(
    image_3hw: torch.Tensor, attribution_hw: torch.Tensor
) -> plt.Figure:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image_3hw.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    attr = attribution_hw.cpu().numpy()
    attr_norm = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("Input")
    axes[0].axis("off")
    axes[1].imshow(img)
    axes[1].imshow(attr_norm, alpha=0.6, cmap="hot")
    axes[1].set_title("Attribution")
    axes[1].axis("off")
    fig.tight_layout()
    return fig


class CIAOFunnyBirdsExplainer(AbstractAttributionExplainer):
    """CIAO search-algorithm adapter for FunnyBirds evaluation.

    Runs CIAO's pipeline (segment → score → search) and converts the list of
    found regions into a [1, 1, H, W] attribution map.  Each region's pixel
    mask is weighted by that region's log-odds drop score.

    Args:
        model:          FunnyBirds model (nn.Module, already on the target device).
        class_names:    List of class name strings, length == num_classes.
        method:         Search algorithm.  Currently only 'lookahead' is
                        available on this branch; other methods can be added
                        after merging feat/mcts-algorithm etc.
        segmentation:   'hex' or 'square'.
        hex_radius:     Circumradius of each hexagon in pixels (used when
                        segmentation='hex').  Default 8 gives ~395 segments
                        for a 256x256 image, fine enough to resolve individual
                        bird parts.
        square_size:    Edge length of each square patch (used when
                        segmentation='square').
        max_regions:    Maximum number of non-overlapping regions to find.
        desired_length: Target number of segments per region.  Tune to cover
                        roughly one bird part (8-30 segments with hex_radius=8).
        ciao_batch_size: Batch size for CIAO's internal forward-pass loops.
    """

    _METHOD_FACTORIES: ClassVar[dict[str, ExplanationMethodFn]] = {
        "lookahead": make_lookahead_method(lookahead_distance=2),
        "mcts": make_mcts_method(num_evals=6400, num_rollouts=64),
        "mcgs": make_mcgs_method(num_evals=6400, num_rollouts=64),
        "ucb": make_ucb_method(step_budget=64, batch_size=16),
        "potential": make_potential_method(step_budget=10),
        "pure_monte_carlo": make_pure_monte_carlo_method(num_evals=100),
        "beam_search": make_beam_search_method(beam_width=64),
    }

    def __init__(
        self,
        model: torch.nn.Module,
        class_names: list[str],
        method: str | ExplanationMethodFn = "lookahead",
        segmentation: str = "hex",
        hex_radius: int = 8,
        square_size: int = 8,
        max_regions: int = 5,
        desired_length: int = 30,
        ciao_batch_size: int = 16,
        mlflow_enabled: bool = False,
    ) -> None:
        # Do not call super().__init__() — we have no Captum explainer object.
        self.explainer = None
        self.baseline = None

        self.predictor = ModelPredictor(model, class_names)

        if segmentation == "hex":
            self.segmentation_fn = make_hexagonal_segmentation(hex_radius=hex_radius)
        elif segmentation == "square":
            self.segmentation_fn = make_square_segmentation(square_size=square_size)
        else:
            raise ValueError(
                f"Unknown segmentation {segmentation!r}. Choose 'hex' or 'square'."
            )

        if callable(method):
            self.method_fn = method
            self.explainer_name = f"CIAO-{method.__name__}"
        elif method in self._METHOD_FACTORIES:
            self.method_fn = self._METHOD_FACTORIES[method]
            self.explainer_name = f"CIAO-{method}"
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                f"Pass an ExplanationMethodFn or one of: {list(self._METHOD_FACTORIES)}."
            )

        self.max_regions = max_regions
        self.desired_length = desired_length
        self.ciao_batch_size = ciao_batch_size

        # Simple single-entry cache so that the two internal explain() calls
        # from get_important_parts → explain() + get_part_importance → explain()
        # only run the expensive CIAO pipeline once per sample.
        self._cache_key: tuple[int, int | None] | None = None
        self._cached_attribution: torch.Tensor | None = None

        self._mlflow_enabled = mlflow_enabled
        self._sample_count = 0

    # ------------------------------------------------------------------
    # Core explain method
    # ------------------------------------------------------------------

    def explain(
        self,
        input: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run CIAO and return a region-weighted attribution map.

        Args:
            input:  [1, 3, H, W] image tensor, values in [0, 1].
            target: Class index as a scalar tensor, or None to use top-1.

        Returns:
            [1, 1, H, W] tensor.  Pixels inside a found region get that
            region's log-odds drop as their value; all other pixels are 0.
        """
        cache_key = (id(input), None if target is None else int(target.item()))
        if cache_key == self._cache_key and self._cached_attribution is not None:
            return self._cached_attribution

        start_time = time.perf_counter()

        assert input.shape[0] == 1, "FunnyBirds evaluation requires batch size 1."
        input_3d = input.squeeze(0)  # [3, H, W]
        H, W = input_3d.shape[1], input_3d.shape[2]
        device = input.device

        target_class_idx = None if target is None else int(target.item())
        replacement_image = _replacement_fn(input_3d)
        image_graph = self.segmentation_fn(input_3d)

        original_logits = self.predictor.get_logits(input)
        if target_class_idx is None:
            target_class_idx = int(original_logits.argmax(dim=1)[0].item())

        original_log_odds_tensor = log_odds_for_class(
            original_logits, target_class_idx
        )[0]
        original_prob = float(
            torch.softmax(original_logits, dim=1)[0, target_class_idx].item()
        )

        X, y = create_surrogate_dataset(
            predictor=self.predictor,
            input_batch=input,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            original_log_odds=original_log_odds_tensor,
            batch_size=self.ciao_batch_size,
        )
        segment_scores = calculate_segment_scores(X, y)

        regions = build_all_regions(
            method=self.method_fn,
            predictor=self.predictor,
            input_batch=input,
            replacement_image=replacement_image,
            image_graph=image_graph,
            target_class_idx=target_class_idx,
            original_log_odds=original_log_odds_tensor,
            scores=segment_scores,
            max_regions=self.max_regions,
            original_prob=original_prob,
            desired_length=self.desired_length,
            batch_size=self.ciao_batch_size,
        )

        # Accumulate weighted masks from all positively-scored regions.
        segments_map = image_graph.segments  # [H, W] int32
        attribution = torch.zeros(1, 1, H, W, device=device)
        for region_result in regions:
            if region_result.score <= 0:
                continue
            mask = torch.zeros(H, W, device=device)
            for seg_id in region_result.region:
                mask[segments_map == seg_id] = 1.0
            attribution[0, 0] += mask * region_result.score

        self._cache_key = cache_key
        self._cached_attribution = attribution

        if self._mlflow_enabled and mlflow.active_run() is not None:
            elapsed = time.perf_counter() - start_time
            with mlflow.start_run(
                run_name=f"sample_{self._sample_count}", nested=True
            ):
                mlflow.log_metrics(
                    {
                        "target_class_idx": float(target_class_idx),
                        "original_log_odds": float(original_log_odds_tensor.item()),
                        "time_seconds": elapsed,
                    }
                )
                for idx, rr in enumerate(regions):
                    mlflow.log_metrics(
                        {
                            f"region_{idx}/score": rr.score,
                            f"region_{idx}/size_segments": float(len(rr.region)),
                        }
                    )
                fig = _make_explanation_figure(input_3d, attribution[0, 0])
                mlflow.log_figure(fig, "figures/explanation.png")
                plt.close(fig)
            self._sample_count += 1

        return attribution

    # ------------------------------------------------------------------
    # Part importance helpers (override to avoid double explain() calls)
    # ------------------------------------------------------------------

    def get_part_importance(
        self,
        image: torch.Tensor,
        part_map: torch.Tensor,
        target: torch.Tensor,
        colors_to_part: dict[tuple[int, int, int], str],
        with_bg: bool = False,
    ) -> dict[str, float]:
        """Sum the CIAO attribution map over each dilated part region."""
        assert image.shape[0] == 1
        attribution = self.explain(image, target=target)

        dilation = nn.MaxPool2d(5, stride=1, padding=2)
        part_importances: dict[str, float] = {}

        for part_color, part_name in colors_to_part.items():
            color_tensor = torch.zeros(1, 3, 1, 1, device=image.device)
            color_tensor[0, 0, 0, 0] = part_color[0]
            color_tensor[0, 1, 0, 0] = part_color[1]
            color_tensor[0, 2, 0, 0] = part_color[2]
            color_mask = torch.all(
                part_map == color_tensor, dim=1, keepdim=True
            ).float()
            color_mask_dilated = dilation(color_mask)
            part_attr = float((attribution * color_mask_dilated).sum().item())

            # Collapse variant suffixes (e.g. 'eye01', 'eye02' → 'eye').
            clean_name = "".join(c for c in part_name if c.isalpha())
            if clean_name in part_importances:
                part_importances[clean_name] += part_attr
            else:
                part_importances[clean_name] = part_attr

        if with_bg:
            for i in range(50):
                color_tensor = torch.zeros(1, 3, 1, 1, device=image.device)
                color_tensor[0, 0, 0, 0] = 204
                color_tensor[0, 1, 0, 0] = 204
                color_tensor[0, 2, 0, 0] = 204 + i
                color_mask = torch.all(
                    part_map == color_tensor, dim=1, keepdim=True
                ).float()
                color_mask_dilated = dilation(color_mask)
                part_attr = float((attribution * color_mask_dilated).sum().item())
                part_importances[f"bg_{str(i).zfill(3)}"] = part_attr

        return part_importances

    def get_important_parts(
        self,
        image: torch.Tensor,
        part_map: torch.Tensor,
        target: torch.Tensor,
        colors_to_part: dict[tuple[int, int, int], str],
        thresholds: np.ndarray,
        with_bg: bool = False,
    ) -> list[list[str]]:
        """Return per-threshold lists of important parts.

        A part is considered important at threshold t if its attribution sum
        exceeds t x total_attribution.
        """
        attribution = self.explain(image, target=target)
        part_importances = self.get_part_importance(
            image, part_map, target, colors_to_part, with_bg=with_bg
        )
        total = float(attribution.sum().item())

        return [
            [key for key, val in part_importances.items() if val > total * threshold]
            for threshold in thresholds
        ]
