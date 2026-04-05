import hydra
import mlflow
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

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

        print(f"Starting explanation for: {cfg.image_path}")

        results = explainer.explain(
            image_path=cfg.image_path,
            predictor=predictor,
            target_class_idx=cfg.target_class_idx,
            max_regions=cfg.max_regions,
            desired_length=cfg.desired_length,
            batch_size=cfg.batch_size,
            segmentation=segmentation,
            method=method,
            replacement=replacement,
        )

        print(
            f"Successfully finished explanation! Found {len(results.regions)} hyperpixels."
        )
        print("\nHyperpixel[0] score: ", results.regions[0].score)
        print(results.class_name)


if __name__ == "__main__":
    main()
