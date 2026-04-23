import time
from contextlib import nullcontext

import hydra
import mlflow
import torch
from hydra.utils import instantiate
from mlflow.entities import Metric
from omegaconf import DictConfig, OmegaConf

from ciao.data.loader import iter_image_paths
from ciao.explainer.ciao_explainer import CIAOExplainer
from ciao.model.predictor import ModelPredictor


MLFLOW_LOG_BATCH_LIMIT = 1000


def _flatten_params(obj: object, parent_key: str = "") -> dict[str, object]:
    """Flatten a nested dict/list into dot/bracket-separated keys for MLflow params."""
    items: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            items.update(_flatten_params(v, new_key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(_flatten_params(v, f"{parent_key}[{i}]"))
    else:
        items[parent_key] = obj
    return items


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment_name)

    with mlflow.start_run(run_name=cfg.logger.run_name):
        params = _flatten_params(OmegaConf.to_container(cfg, resolve=True))
        params.pop("target_class_idx", None)
        mlflow.log_params(params)

        segmentation = instantiate(cfg.segmentation)
        method = instantiate(cfg.method)
        replacement = instantiate(cfg.replacement)

        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)

        explainer = CIAOExplainer()

        batch_mode = cfg.data.get("batch_path") is not None

        for image_path in iter_image_paths(cfg):
            print(f"Starting explanation for: {image_path}")
            start_time = time.perf_counter()

            run_ctx = (
                mlflow.start_run(run_name=image_path.name, nested=True)
                if batch_mode
                else nullcontext(mlflow.active_run())
            )
            with run_ctx as run:
                assert run is not None
                results = explainer.explain(
                    image_path=image_path,
                    predictor=predictor,
                    target_class_idx=cfg.target_class_idx,
                    max_regions=cfg.max_regions,
                    desired_length=cfg.desired_length,
                    batch_size=cfg.batch_size,
                    segmentation=segmentation,
                    method=method,
                    replacement=replacement,
                )

                elapsed = time.perf_counter() - start_time

                mlflow.log_params(
                    {
                        "target_class_idx": results.target_class_idx,
                        "class_name": results.class_name,
                    }
                )

                # Log per-region summary metrics
                for idx, region in enumerate(results.regions):
                    mlflow.log_metrics(
                        {
                            f"region_{idx}/final_score": region.score,
                            f"region_{idx}/original_prob": region.original_prob,
                            f"region_{idx}/masked_prob": region.masked_prob,
                            f"region_{idx}/probability_drop": region.probability_drop,
                            f"region_{idx}/evaluations_count": region.evaluations_count,
                        }
                    )

                # Batch-log trajectory points for graphs
                ts_ms = int(time.time() * 1000)
                trajectory_metrics = [
                    Metric(
                        key=f"region_{idx}/trajectory_best_score",
                        value=item["best_score"],
                        timestamp=ts_ms,
                        step=int(item["evals"]),
                    )
                    for idx, region in enumerate(results.regions)
                    for item in region.trajectory
                ]
                if trajectory_metrics:
                    client = mlflow.MlflowClient()
                    for start in range(
                        0, len(trajectory_metrics), MLFLOW_LOG_BATCH_LIMIT
                    ):
                        client.log_batch(
                            run_id=run.info.run_id,
                            metrics=trajectory_metrics[
                                start : start + MLFLOW_LOG_BATCH_LIMIT
                            ],
                        )

                mlflow.log_metric("time_seconds", elapsed)

                if not results.regions:
                    print(
                        f"Done: {image_path.name} | no regions found | time={elapsed:.1f}s"
                    )
                    continue

                best_region = results.regions[0]
                print(
                    f"Done: {image_path.name} | "
                    f"score={best_region.score:.4f} | "
                    f"prob_drop={best_region.probability_drop:.4f} | "
                    f"evals={best_region.evaluations_count} | "
                    f"time={elapsed:.1f}s"
                )


if __name__ == "__main__":
    main()
