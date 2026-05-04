"""Compute per-algorithm saliency maps from an MLflow experiment and optionally evaluate deletion AUC.

Logs results back to the existing nested runs.

Usage (Hydra CLI):
    uv run python tools/compute_saliency.py \
        images_path=/mnt/projects/explainability/ciao/imagenet_s/comparison/images \
        experiment_name=algorithm-comparison-imagenet-s \
        compute_deletion=true

The script expects that the explanation runs were produced with segments.npy
artifacts (requires __main__.py from this branch). Runs missing that artifact
are skipped with a warning.
"""

import re
import tempfile
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
from ciao.data.preprocessing import load_and_preprocess_image as _default_preprocess
from ciao.metrics import (
    build_saliency_map,
    compute_deletion_curve,
    compute_insertion_curve,
    compute_iou_top_fraction,
    compute_pointing_game,
)
from ciao.model.predictor import ModelPredictor
from ciao.visualization import (  # type: ignore[attr-defined]  # plot_insertion_curve added in feat/baselines
    plot_deletion_curve,
    plot_insertion_curve,
    plot_saliency_map,
)


# Run name format produced by the job script: {label}-{seed}-length-{desired_length}
_RUN_NAME_RE = re.compile(r"^(.+)-(\d+)-length-(\d+)$")


def _parse_parent_run_name(run_name: str) -> tuple[str, str] | None:
    """Return (algorithm_label, seed) by stripping the length suffix, or None."""
    m = _RUN_NAME_RE.match(run_name)
    if m is None:
        return None
    return m.group(1), m.group(2)


def _load_segments_npy(client: mlflow.MlflowClient, run_id: str) -> np.ndarray | None:
    """Download segments.npy artifact and return as numpy array, or None if missing."""
    try:
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        if "segments.npy" not in artifacts:
            return None
        buf = client.download_artifacts(run_id, "segments.npy")
        return np.load(buf)
    except Exception:
        return None


def _load_region_segments(
    client: mlflow.MlflowClient, run_id: str
) -> list[list[int]] | None:
    """Load all region_X/segments.json artifacts and return list of segment-ID lists."""
    try:
        artifacts = {a.path for a in client.list_artifacts(run_id, path="")}
        regions: list[list[int]] = []
        idx = 0
        while True:
            path = f"region_{idx}/segments.json"
            if path not in artifacts:
                # Try listing the region subdirectory
                sub = f"region_{idx}"
                sub_artifacts = {
                    a.path for a in client.list_artifacts(run_id, path=sub)
                }
                seg_path = f"{sub}/segments.json"
                if seg_path not in sub_artifacts:
                    break
                path = seg_path
            local = client.download_artifacts(run_id, path)
            import json

            with open(local) as f:
                data = json.load(f)
            regions.append(data["segments"])
            idx += 1
        return regions if regions else None
    except Exception:
        return None


def _build_binary_mask(
    segments_npy: np.ndarray, selected_segment_ids: list[int]
) -> np.ndarray:
    """Return a [H, W] bool mask: True where pixels belong to any selected segment."""
    segments = torch.tensor(segments_npy, dtype=torch.int32)
    ids = torch.tensor(selected_segment_ids, dtype=torch.int32)
    return torch.isin(segments, ids).numpy()


@hydra.main(
    version_base=None, config_path="../configs", config_name="saliency_analysis"
)
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.tracking_uri)
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(cfg.experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")

    # ------------------------------------------------------------------
    # 1. Collect all runs in the experiment and build a lookup structure:
    #    algorithm_key -> image_name -> list of child run IDs
    # ------------------------------------------------------------------
    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=50000,
    )

    # Separate parent runs from nested per-image child runs
    parent_by_id: dict[str, mlflow.entities.Run] = {}
    children_by_parent: dict[str, list[mlflow.entities.Run]] = defaultdict(list)

    for run in all_runs:
        parent_id = run.data.tags.get("mlflow.parentRunId")
        if parent_id is None:
            parent_by_id[run.info.run_id] = run
        else:
            children_by_parent[parent_id].append(run)

    # algorithm_key = "{run_label}-{seed}"  (desired_length stripped)
    # Mapping: algorithm_key -> image_name -> {desired_length: child_run_id}
    grouped: dict[str, dict[str, dict[int, str]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for parent_id, parent_run in parent_by_id.items():
        parsed = _parse_parent_run_name(parent_run.info.run_name or "")
        if parsed is None:
            print(
                f"Skipping parent run with unexpected name: {parent_run.info.run_name!r}"
            )
            continue
        algo_key, _seed = parsed
        # Recover desired_length from params (more robust than name parsing)
        desired_length_str = parent_run.data.params.get("desired_length")
        if desired_length_str is None:
            continue
        desired_length = int(desired_length_str)

        for child in children_by_parent.get(parent_id, []):
            image_name = child.info.run_name or ""
            grouped[algo_key][image_name][desired_length] = child.info.run_id

    # ------------------------------------------------------------------
    # 2. Optionally set up model + replacement for deletion
    # ------------------------------------------------------------------
    predictor: ModelPredictor | None = None
    replacement_fn = None

    preprocess_fn = (
        instantiate(cfg.preprocessing)
        if "preprocessing" in cfg
        else _default_preprocess
    )

    if cfg.compute_deletion or cfg.compute_insertion:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate(cfg.model).to(device)
        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)
        replacement_fn = instantiate(cfg.replacement)
        print(f"Deletion/insertion enabled — using device: {device}")

    images_path = Path(cfg.images_path)

    masks_path = Path(cfg.masks_path) if cfg.masks_path else None
    imagenet_s_mapping = build_imagenet_s_mapping() if masks_path is not None else None

    # ------------------------------------------------------------------
    # 3. For each (algorithm, image): build saliency map, compute deletion,
    #    log back to every child run in that group.
    # ------------------------------------------------------------------
    if cfg.get("algo_filter"):
        grouped = {k: v for k, v in grouped.items() if k.startswith(cfg.algo_filter)}

    total_groups = sum(len(imgs) for imgs in grouped.values())
    done = 0

    for algo_key, images in grouped.items():
        for image_name, length_to_run_id in images.items():
            done += 1
            print(f"[{done}/{total_groups}] {algo_key} / {image_name}")

            # Collect binary masks across desired_lengths
            binary_masks: list[np.ndarray] = []
            segments_npy: np.ndarray | None = None
            target_class_idx: int | None = None

            for desired_length, run_id in sorted(length_to_run_id.items()):
                seg = _load_segments_npy(client, run_id)
                if seg is None:
                    print(
                        f"  WARN: segments.npy missing for run {run_id} "
                        f"(length={desired_length}) — skipping this length"
                    )
                    continue

                if segments_npy is None:
                    segments_npy = seg

                region_segs = _load_region_segments(client, run_id)
                if region_segs is None:
                    print(
                        f"  WARN: no region segments found for run {run_id} "
                        f"(length={desired_length}) — skipping this length"
                    )
                    continue

                # Union of all regions → single binary mask for this run
                all_ids = [sid for region in region_segs for sid in region]
                if all_ids:
                    binary_masks.append(_build_binary_mask(seg, all_ids))

                # Recover target class from child run params
                if target_class_idx is None:
                    tc = client.get_run(run_id).data.params.get("target_class_idx")
                    if tc is not None:
                        target_class_idx = int(tc)

            if not binary_masks:
                print(f"  WARN: no usable masks for {algo_key}/{image_name} — skipping")
                continue

            saliency_map = build_saliency_map(
                binary_masks, sigma_fraction=cfg.sigma_fraction
            )

            # Compute pointing game if masks are available
            pointing_game: bool | None = None
            if (
                masks_path is not None
                and imagenet_s_mapping is not None
                and target_class_idx is not None
            ):
                mask_file = masks_path / (Path(image_name).stem + ".png")
                if mask_file.exists():
                    pointing_game = compute_pointing_game(
                        saliency_map=saliency_map,
                        target_class_idx=target_class_idx,
                        gt_mask=load_mask(mask_file),
                        mapping=imagenet_s_mapping,
                    )

            # Compute deletion/insertion curves + figures (requires model + images)
            deletion_auc: float | None = None
            deletion_fractions: np.ndarray | None = None
            deletion_probs: np.ndarray | None = None
            insertion_auc: float | None = None
            insertion_fractions: np.ndarray | None = None
            insertion_probs: np.ndarray | None = None
            iou: float | None = None
            input_tensor: torch.Tensor | None = None
            class_name: str | None = None

            needs_image = (
                cfg.compute_deletion or cfg.compute_insertion or cfg.log_figures
            )
            if needs_image and predictor is not None and replacement_fn is not None:
                image_path = images_path / image_name
                if not image_path.exists():
                    print(
                        f"  WARN: image not found at {image_path} — skipping deletion/figures"
                    )
                else:
                    input_tensor = preprocess_fn(image_path, device=predictor.device)
                    replacement_image = replacement_fn(input_tensor)
                    input_batch = input_tensor.unsqueeze(0)

                    if target_class_idx is None:
                        target_class_idx = int(
                            predictor.get_logits(input_batch).argmax(dim=1)[0].item()
                        )

                    class_name = predictor.class_names[target_class_idx]

                    if cfg.compute_deletion:
                        deletion_fractions, deletion_probs = compute_deletion_curve(
                            image=input_batch,
                            saliency_map=saliency_map,
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
                            saliency_map=saliency_map,
                            target_class_idx=target_class_idx,
                            predictor=predictor,
                            replacement_image=replacement_image,
                            n_steps=cfg.insertion_steps,
                        )
                        insertion_auc = float(
                            np.trapezoid(insertion_probs, insertion_fractions)
                        )

            # Compute IoU if masks and top_fraction are configured
            if (
                cfg.iou_top_fraction is not None
                and masks_path is not None
                and imagenet_s_mapping is not None
                and target_class_idx is not None
            ):
                mask_file = masks_path / (Path(image_name).stem + ".png")
                if mask_file.exists():
                    gt_mask = load_mask(mask_file)
                    object_mask = get_object_mask(
                        gt_mask, target_class_idx, imagenet_s_mapping
                    )
                    if object_mask is not None:
                        H, W = saliency_map.shape
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
                            saliency_map=saliency_map,
                            gt_mask=gt_resized,
                            top_fraction=cfg.iou_top_fraction,
                        )

            # Log back to every child run in this group
            with tempfile.TemporaryDirectory() as _tmpdir:
                tmp = Path(_tmpdir)
                saliency_path = tmp / "saliency_map.npy"
                np.save(saliency_path, saliency_map)

                for run_id in length_to_run_id.values():
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_artifact(str(saliency_path))
                        if deletion_auc is not None:
                            mlflow.log_metric("deletion_auc", deletion_auc)
                        if insertion_auc is not None:
                            mlflow.log_metric("insertion_auc", insertion_auc)
                        if pointing_game is not None:
                            mlflow.log_metric("pointing_game", float(pointing_game))
                        if iou is not None:
                            mlflow.log_metric("iou", iou)
                        if cfg.log_figures and input_tensor is not None:
                            fig_sal = plot_saliency_map(
                                input_tensor, saliency_map, class_name
                            )
                            mlflow.log_figure(fig_sal, "figures/saliency_map.png")
                            plt.close(fig_sal)
                        if (
                            cfg.log_figures
                            and deletion_fractions is not None
                            and deletion_probs is not None
                        ):
                            fig_del = plot_deletion_curve(
                                deletion_fractions,
                                deletion_probs,
                                deletion_auc,
                                class_name,
                            )
                            mlflow.log_figure(fig_del, "figures/deletion_curve.png")
                            plt.close(fig_del)
                        if (
                            cfg.log_figures
                            and insertion_fractions is not None
                            and insertion_probs is not None
                        ):
                            fig_ins = plot_insertion_curve(
                                insertion_fractions,
                                insertion_probs,
                                insertion_auc,
                                class_name,
                            )
                            mlflow.log_figure(fig_ins, "figures/insertion_curve.png")
                            plt.close(fig_ins)

            parts = [f"masks={len(binary_masks)}"]
            if deletion_auc is not None:
                parts.append(f"deletion_auc={deletion_auc:.4f}")
            if insertion_auc is not None:
                parts.append(f"insertion_auc={insertion_auc:.4f}")
            if pointing_game is not None:
                parts.append(f"pointing_game={'hit' if pointing_game else 'miss'}")
            if iou is not None:
                parts.append(f"iou={iou:.4f}")
            print(f"  Done — {', '.join(parts)}")

    print(f"\nFinished. Processed {done} (algorithm, image) groups.")


if __name__ == "__main__":
    main()
