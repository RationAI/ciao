"""Entrypoint for running Meaningful Perturbations explanations over a batch of images."""

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
from ciao.baselines.meaningful_perturbations.explainer import (
    MeaningfulPerturbationsExplainer,
)
from ciao.data.loader import iter_image_paths
from ciao.model.predictor import ModelPredictor


@hydra.main(version_base=None, config_path="../../../configs", config_name="mp_base")
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

        preprocess_fn = (
            instantiate(cfg.preprocessing) if "preprocessing" in cfg else None
        )
        explainer = MeaningfulPerturbationsExplainer()
        batch_mode = cfg.data.get("batch_path") is not None

        for image_path in iter_image_paths(cfg):
            print(f"[MP] Starting explanation for: {image_path}")
            start_time = time.perf_counter()

            run_ctx = (
                mlflow.start_run(run_name=image_path.name, nested=True)
                if batch_mode
                else nullcontext(mlflow.active_run())
            )
            with run_ctx as run:
                assert run is not None
                results, mp_result = explainer.explain(
                    image_path=image_path,
                    predictor=predictor,
                    replacement=replacement,
                    target_class_idx=cfg.target_class_idx,
                    area=cfg.mp.area,
                    area_lambda=cfg.mp.area_lambda,
                    area_lambda_growth=cfg.mp.area_lambda_growth,
                    tv_lambda=cfg.mp.tv_lambda,
                    tv_beta=cfg.mp.tv_beta,
                    max_time=cfg.mp.max_time,
                    max_iterations=cfg.mp.max_iterations,
                    learning_rate=cfg.mp.learning_rate,
                    mask_step=cfg.mp.mask_step,
                    mask_sigma=cfg.mp.mask_sigma,
                    jitter=cfg.mp.jitter,
                    jitter_tau=cfg.mp.jitter_tau,
                    batch_size=cfg.batch_size,
                    preprocess_fn=preprocess_fn,
                )

                elapsed = time.perf_counter() - start_time

                extra_metrics = {
                    "mp/final_loss": mp_result.final_loss,
                    "mp/final_target_logprob": mp_result.final_target_logprob,
                    "mp/final_area": mp_result.final_area,
                    "mp/iterations": float(mp_result.iterations),
                }

                log_explanation_results(
                    run.info.run_id,
                    results,
                    elapsed,
                    extra_metrics=extra_metrics,
                )
                if cfg.logger.log_figures:
                    log_figures(results)
                print_summary(image_path, results, elapsed, method_label="MP")


if __name__ == "__main__":
    main()
