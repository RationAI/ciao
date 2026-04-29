"""Shared CLI helpers for baseline explainers (extremal perturbations, LIME)."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
from mlflow.entities import Metric

from ciao.visualization import plot_overview, plot_region_scores, plot_regions


if TYPE_CHECKING:
    from pathlib import Path

    from ciao.explainer.ciao_explainer import ExplanationResult


MLFLOW_LOG_BATCH_LIMIT = 1000


def seed_everything(seed: int) -> None:
    """Seed all RNG sources used (or potentially used) downstream."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_params(obj: object, parent_key: str = "") -> dict[str, object]:
    """Flatten a nested dict/list into dot/bracket-separated keys for MLflow params."""
    items: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            items.update(flatten_params(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten_params(v, f"{parent_key}.{i}"))
    else:
        items[parent_key] = obj
    return items


def log_trajectory(run_id: str, results: ExplanationResult) -> None:
    """Batch-log per-region trajectory points to MLflow."""
    ts_ms = int(time.time() * 1000)
    trajectory_metrics = [
        metric
        for idx, region in enumerate(results.regions)
        for item in {item["evals"]: item for item in region.trajectory}.values()
        for metric in (
            Metric(
                key=f"region_{idx}/trajectory_best_score",
                value=item["best_score"],
                timestamp=ts_ms,
                step=int(item["evals"]),
            ),
            Metric(
                key=f"region_{idx}/trajectory_time",
                value=item["time"],
                timestamp=ts_ms,
                step=int(item["evals"]),
            ),
        )
    ]
    if not trajectory_metrics:
        return

    client = mlflow.MlflowClient()
    for start in range(0, len(trajectory_metrics), MLFLOW_LOG_BATCH_LIMIT):
        client.log_batch(
            run_id=run_id,
            metrics=trajectory_metrics[start : start + MLFLOW_LOG_BATCH_LIMIT],
        )


def log_explanation_results(
    run_id: str,
    results: ExplanationResult,
    elapsed: float,
    extra_metrics: dict[str, float] | None = None,
    extra_params: dict[str, object] | None = None,
) -> None:
    """Log explanation params, baseline log-odds, per-region metrics, trajectory, and timing to MLflow."""
    mlflow.log_params(
        {
            "target_class_idx": results.target_class_idx,
            "class_name": results.class_name,
        }
    )
    mlflow.log_metric("original_log_odds", results.original_log_odds)

    for idx, region in enumerate(results.regions):
        mlflow.log_metrics(
            {
                f"region_{idx}/final_score": region.score,
                f"region_{idx}/original_prob": region.original_prob,
                f"region_{idx}/masked_prob": region.masked_prob,
                f"region_{idx}/probability_drop": region.probability_drop,
                f"region_{idx}/evaluations_count": region.evaluations_count,
                f"region_{idx}/masked_top_prob": region.masked_top_prob,
            }
        )
        mlflow.log_params(
            {
                f"region_{idx}/masked_top_class_idx": region.masked_top_class_idx,
                f"region_{idx}/masked_top_class_name": region.masked_top_class_name,
            }
        )
        mlflow.log_dict(
            {"segments": sorted(region.region)},
            f"region_{idx}/segments.json",
        )

    log_trajectory(run_id, results)

    if extra_metrics:
        mlflow.log_metrics(extra_metrics)
    if extra_params:
        mlflow.log_params(extra_params)

    mlflow.log_metric("time_seconds", elapsed)


def log_figures(results: ExplanationResult) -> None:
    """Render visualization figures and log them as MLflow artifacts."""
    if not results.regions:
        return

    for name, plot_fn in (
        ("overview", plot_overview),
        ("regions", plot_regions),
        ("region_scores", plot_region_scores),
    ):
        fig = plot_fn(results)
        mlflow.log_figure(fig, f"figures/{name}.png")
        plt.close(fig)


def print_summary(
    image_path: Path,
    results: ExplanationResult,
    elapsed: float,
    method_label: str,
) -> None:
    """Print a one-line summary of the explanation outcome for an image."""
    if not results.regions:
        print(
            f"[{method_label}] Done: {image_path.name} | "
            f"class={results.class_name} (idx={results.target_class_idx}) | "
            f"no regions found | time={elapsed:.1f}s"
        )
        return

    best = results.regions[0]
    print(
        f"[{method_label}] Done: {image_path.name} | "
        f"class={results.class_name} (idx={results.target_class_idx}) | "
        f"score={best.score:.4f}, "
        f"prob {best.original_prob:.4f} -> {best.masked_prob:.4f} "
        f"(drop={best.probability_drop:.4f}), "
        f"size={len(best.region)} segs, evals={best.evaluations_count}, "
        f"shift->{best.masked_top_class_name} ({best.masked_top_prob:.4f}) | "
        f"time={elapsed:.1f}s"
    )
