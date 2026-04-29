"""Entrypoint for running extremal-perturbation explanations over a batch of images."""

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
from ciao.baselines.extremal.explainer import ExtremalPerturbationExplainer
from ciao.data.loader import iter_image_paths
from ciao.model.predictor import ModelPredictor


@hydra.main(version_base=None, config_path="../../../configs", config_name="ep_base")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment_name)

    with mlflow.start_run(run_name=cfg.logger.run_name):
        params = flatten_params(OmegaConf.to_container(cfg, resolve=True))
        params.pop("target_class_idx", None)
        mlflow.log_params(params)

        replacement = instantiate(cfg.replacement)
        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)

        explainer = ExtremalPerturbationExplainer()

        batch_mode = cfg.data.get("batch_path") is not None

        for image_path in iter_image_paths(cfg):
            print(f"[EP] Starting explanation for: {image_path}")
            start_time = time.perf_counter()

            run_ctx = (
                mlflow.start_run(run_name=image_path.name, nested=True)
                if batch_mode
                else nullcontext(mlflow.active_run())
            )
            with run_ctx as run:
                assert run is not None
                results, ep_result = explainer.explain(
                    image_path=image_path,
                    predictor=predictor,
                    replacement=replacement,
                    target_class_idx=cfg.target_class_idx,
                    area=cfg.ep.area,
                    max_time=cfg.ep.max_time,
                    max_iterations=cfg.ep.max_iterations,
                    learning_rate=cfg.ep.learning_rate,
                    mask_step=cfg.ep.mask_step,
                    mask_sigma=cfg.ep.mask_sigma,
                    area_lambda=cfg.ep.area_lambda,
                    area_lambda_growth=cfg.ep.area_lambda_growth,
                    batch_size=cfg.batch_size,
                )

                elapsed = time.perf_counter() - start_time

                extra_metrics = {
                    "ep/final_loss": ep_result.final_loss,
                    "ep/final_target_logprob": ep_result.final_target_logprob,
                    "ep/final_area": ep_result.final_area,
                    "ep/iterations": float(ep_result.iterations),
                }

                log_explanation_results(
                    run.info.run_id,
                    results,
                    elapsed,
                    extra_metrics=extra_metrics,
                )
                if cfg.logger.log_figures:
                    log_figures(results)
                print_summary(image_path, results, elapsed, method_label="EP")


if __name__ == "__main__":
    main()
