import time

import hydra
import mlflow
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from ciao.data.loader import iter_image_paths
from ciao.explainer.ciao_explainer import CIAOExplainer
from ciao.model.predictor import ModelPredictor


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.logger.tracking_uri)
    mlflow.set_experiment(cfg.logger.experiment_name)

    with mlflow.start_run(run_name=cfg.logger.run_name):
        params = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(params)  # type: ignore[arg-type]

        segmentation = instantiate(cfg.segmentation)
        method = instantiate(cfg.method)
        replacement = instantiate(cfg.replacement)

        model = instantiate(cfg.model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        class_names = instantiate(cfg.classes)
        predictor = ModelPredictor(model=model, class_names=class_names)

        explainer = CIAOExplainer()

        for image_path in iter_image_paths(cfg):
            print(f"Starting explanation for: {image_path}")
            start_time = time.perf_counter()

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

            # Log metrics for every region
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

                # Log trajectory for graphs
                for item in region.trajectory:
                    mlflow.log_metric(
                        f"region_{idx}/trajectory_best_score",
                        item["best_score"],
                        step=int(item["evals"]),
                    )

            mlflow.log_metric("time_seconds", elapsed)

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
