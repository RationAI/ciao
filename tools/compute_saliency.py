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
import mlflow
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ciao.data.preprocessing import load_and_preprocess_image
from ciao.metrics import build_saliency_map, compute_deletion_auc
from ciao.model.predictor import ModelPredictor


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
        algo_key, seed = parsed
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

    if cfg.compute_deletion:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = instantiate(cfg.model).to(device)
        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)
        replacement_fn = instantiate(cfg.replacement)
        print(f"Deletion enabled — using device: {device}")

    images_path = Path(cfg.images_path)

    # ------------------------------------------------------------------
    # 3. For each (algorithm, image): build saliency map, compute deletion,
    #    log back to every child run in that group.
    # ------------------------------------------------------------------
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

            # Compute deletion AUC if requested
            deletion_auc: float | None = None
            if (
                cfg.compute_deletion
                and predictor is not None
                and replacement_fn is not None
            ):
                image_path = images_path / image_name
                if not image_path.exists():
                    print(
                        f"  WARN: image not found at {image_path} — skipping deletion"
                    )
                else:
                    input_tensor = load_and_preprocess_image(
                        image_path, device=predictor.device
                    )
                    replacement_image = replacement_fn(input_tensor)
                    input_batch = input_tensor.unsqueeze(0)

                    if target_class_idx is None:
                        target_class_idx = int(
                            predictor.get_logits(input_batch).argmax(dim=1)[0].item()
                        )

                    deletion_auc = compute_deletion_auc(
                        image=input_batch,
                        saliency_map=saliency_map,
                        target_class_idx=target_class_idx,
                        predictor=predictor,
                        replacement_image=replacement_image,
                        n_steps=cfg.deletion_steps,
                    )

            # Log back to every child run in this group
            with tempfile.TemporaryDirectory() as tmpdir:
                saliency_path = Path(tmpdir) / "saliency_map.npy"
                np.save(saliency_path, saliency_map)

                for run_id in length_to_run_id.values():
                    with mlflow.start_run(run_id=run_id):
                        mlflow.log_artifact(str(saliency_path))
                        if deletion_auc is not None:
                            mlflow.log_metric("deletion_auc", deletion_auc)

            status = (
                f"deletion_auc={deletion_auc:.4f}"
                if deletion_auc is not None
                else "no deletion"
            )
            print(f"  Done — masks={len(binary_masks)}, {status}")

    print(f"\nFinished. Processed {done} (algorithm, image) groups.")


if __name__ == "__main__":
    main()
