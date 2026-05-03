r"""Compute deletion/insertion AUC and IoU for baseline explanation runs in MLflow.

Reads baseline runs (GradCAM, GradCAM++, Occlusion, MP, LIME) from an MLflow
experiment, loads the ``soft_mask.npy`` artifact saved by the baseline CLI, and
computes deletion AUC, insertion AUC, and optionally IoU. Results are logged back
to the existing per-image child runs.

Usage (Hydra CLI):
    uv run python tools/compute_baselines_saliency.py \
        images_path=/mnt/projects/explainability/ciao/imagenet_s/comparison/images \
        experiment_name=baseline-comparison \
        compute_deletion=true \
        compute_insertion=true \
        iou_top_fraction=0.1

Runs missing ``soft_mask.npy`` are skipped with a warning.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ciao.data.imagenet_s import build_imagenet_s_mapping, get_object_mask, load_mask
from ciao.data.preprocessing import load_and_preprocess_image
from ciao.metrics import (  # type: ignore[attr-defined]  # added in feat/metrics
    compute_deletion_curve,
    compute_insertion_curve,
    compute_iou_top_fraction,
)
from ciao.model.predictor import ModelPredictor
from ciao.visualization import plot_deletion_curve, plot_insertion_curve


def _load_soft_mask(client: mlflow.MlflowClient, run_id: str) -> np.ndarray | None:
    """Download soft_mask.npy artifact and return as numpy array, or None if missing."""
    try:
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        if "soft_mask.npy" not in artifacts:
            return None
        local = client.download_artifacts(run_id, "soft_mask.npy")
        return np.load(local)
    except Exception:
        return None


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="baselines_saliency_analysis",
)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=50000,
    )

    # Separate parent runs (batch-level) from child runs (per-image).
    parent_by_id: dict[str, mlflow.entities.Run] = {}
    children_by_parent: dict[str, list[mlflow.entities.Run]] = defaultdict(list)

    for run in all_runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id is None:
            parent_by_id[run.info.run_id] = run
        else:
            children_by_parent[parent_id].append(run)

    # Collect all child runs that have a parent (batch mode) plus any parentless runs.
    image_runs: list[mlflow.entities.Run] = []
    for parent_id in parent_by_id:
        children = children_by_parent.get(parent_id, [])
        if children:
            image_runs.extend(children)
        else:
            # Single-image run (no batch parent/child nesting).
            image_runs.append(parent_by_id[parent_id])

    # Set up model + replacement for deletion / insertion.
    predictor: ModelPredictor | None = None
    replacement_fn = None
    needs_model = cfg.compute_deletion or cfg.compute_insertion

    if needs_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate(cfg.model).to(device)
        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)
        replacement_fn = instantiate(cfg.replacement)
        print(f"Model loaded — using device: {device}")

    images_path = Path(cfg.images_path)
    masks_path = Path(cfg.masks_path) if cfg.get("masks_path") else None
    imagenet_s_mapping = build_imagenet_s_mapping() if masks_path is not None else None

    total = len(image_runs)
    for idx, run in enumerate(image_runs):
        run_id = run.info.run_id
        image_name = run.info.run_name or ""
        print(f"[{idx + 1}/{total}] {image_name} (run {run_id[:8]})")

        soft_mask = _load_soft_mask(client, run_id)
        if soft_mask is None:
            print("  WARN: soft_mask.npy missing — skipping")
            continue

        target_class_idx_str = run.data.params.get("target_class_idx")
        if target_class_idx_str is None:
            print("  WARN: target_class_idx param missing — skipping")
            continue
        target_class_idx = int(target_class_idx_str)
        class_name = run.data.params.get("class_name")

        deletion_auc: float | None = None
        deletion_fractions: np.ndarray | None = None
        deletion_probs: np.ndarray | None = None
        insertion_auc: float | None = None
        insertion_fractions: np.ndarray | None = None
        insertion_probs: np.ndarray | None = None
        iou: float | None = None

        if needs_model and predictor is not None and replacement_fn is not None:
            image_path = images_path / image_name
            if not image_path.exists():
                print(
                    f"  WARN: image not found at {image_path} — skipping model metrics"
                )
            else:
                input_tensor = load_and_preprocess_image(
                    image_path, device=predictor.device
                )
                input_batch = input_tensor.unsqueeze(0)
                replacement_image = replacement_fn(input_tensor)

                if cfg.compute_deletion:
                    deletion_fractions, deletion_probs = compute_deletion_curve(
                        image=input_batch,
                        saliency_map=soft_mask,
                        target_class_idx=target_class_idx,
                        predictor=predictor,
                        replacement_image=replacement_image,
                        n_steps=cfg.deletion_steps,
                    )
                    deletion_auc = float(
                        np.trapezoid(deletion_probs, deletion_fractions)
                    )

                if cfg.compute_insertion:
                    insertion_fractions, insertion_probs = compute_insertion_curve(
                        image=input_batch,
                        saliency_map=soft_mask,
                        target_class_idx=target_class_idx,
                        predictor=predictor,
                        replacement_image=replacement_image,
                        n_steps=cfg.insertion_steps,
                    )
                    insertion_auc = float(
                        np.trapezoid(insertion_probs, insertion_fractions)
                    )

        if (
            masks_path is not None
            and imagenet_s_mapping is not None
            and cfg.get("iou_top_fraction") is not None
        ):
            mask_file = masks_path / (Path(image_name).stem + ".png")
            if mask_file.exists():
                gt_mask_raw = load_mask(mask_file)
                object_mask = get_object_mask(
                    gt_mask_raw, target_class_idx, imagenet_s_mapping
                )
                if object_mask is not None:
                    H, W = soft_mask.shape
                    gt_resized = (
                        torch.nn.functional.interpolate(
                            object_mask.float().unsqueeze(0).unsqueeze(0),
                            size=(H, W),
                            mode="nearest",
                        )
                        .squeeze()
                        .bool()
                        .numpy()
                    )
                    iou = compute_iou_top_fraction(
                        soft_mask, gt_resized, top_fraction=cfg.iou_top_fraction
                    )

        # Log metrics and figures back to the run.
        with mlflow.start_run(run_id=run_id):
            if deletion_auc is not None:
                mlflow.log_metric("deletion_auc", deletion_auc)
            if insertion_auc is not None:
                mlflow.log_metric("insertion_auc", insertion_auc)
            if iou is not None:
                mlflow.log_metric(f"iou_top{cfg.iou_top_fraction}", iou)

            if cfg.log_figures:
                if deletion_fractions is not None and deletion_probs is not None:
                    fig = plot_deletion_curve(
                        deletion_fractions, deletion_probs, deletion_auc, class_name
                    )
                    mlflow.log_figure(fig, "figures/deletion_curve.png")
                    plt.close(fig)
                if insertion_fractions is not None and insertion_probs is not None:
                    fig = plot_insertion_curve(
                        insertion_fractions, insertion_probs, insertion_auc, class_name
                    )
                    mlflow.log_figure(fig, "figures/insertion_curve.png")
                    plt.close(fig)

        parts: list[str] = []
        if deletion_auc is not None:
            parts.append(f"del_auc={deletion_auc:.4f}")
        if insertion_auc is not None:
            parts.append(f"ins_auc={insertion_auc:.4f}")
        if iou is not None:
            parts.append(f"iou={iou:.4f}")
        print(f"  Done — {', '.join(parts) if parts else 'no metrics computed'}")

    print(f"\nFinished. Processed {total} runs.")


if __name__ == "__main__":
    main()
