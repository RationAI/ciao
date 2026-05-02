"""Entrypoint for running Captum-LIME explanations over a batch of images."""

from __future__ import annotations

import time
from contextlib import nullcontext

import hydra
import mlflow
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ciao.baselines._cli import (
    flatten_params,
    log_explanation_results,
    log_figures,
    print_summary,
    seed_everything,
)
from ciao.baselines.lime.explainer import LimeExplainer
from ciao.data.loader import iter_image_paths
from ciao.model.predictor import ModelPredictor


@hydra.main(version_base=None, config_path="../../../configs", config_name="lime_base")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment_name)

    with mlflow.start_run(run_name=cfg.logger.run_name):
        params = flatten_params(OmegaConf.to_container(cfg, resolve=True))
        params.pop("target_class_idx", None)
        mlflow.log_params(params)

        segmentation = instantiate(cfg.segmentation)
        replacement = instantiate(cfg.replacement)
        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)

        explainer = LimeExplainer()

        batch_mode = cfg.data.get("batch_path") is not None

        for image_path in iter_image_paths(cfg):
            print(f"[LIME] Starting explanation for: {image_path}")
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
                    segmentation=segmentation,
                    replacement=replacement,
                    target_class_idx=cfg.target_class_idx,
                    sigma=cfg.sigma,
                    desired_length=cfg.desired_length,
                    n_samples=cfg.lime.n_samples,
                    batch_size=cfg.batch_size,
                )

                elapsed = time.perf_counter() - start_time

                extra_metrics = {"lime/n_samples": float(cfg.lime.n_samples)}

                log_explanation_results(
                    run.info.run_id,
                    results,
                    elapsed,
                    extra_metrics=extra_metrics,
                )
                if cfg.logger.log_figures:
                    log_figures(results)
                print_summary(image_path, results, elapsed, method_label="LIME")


if __name__ == "__main__":
    main()
