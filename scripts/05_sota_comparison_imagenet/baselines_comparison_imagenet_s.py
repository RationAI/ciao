from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/images"
MASKS_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/masks"
EXPERIMENT_NAME = "baselines-comparison-imagenet-s"
BRANCH = "test/all-algorithms"

COMMON = (
    "logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
    "data=imagenet_s_batch "
    f"data.batch_path={IMAGES_PATH} "
    f"data.masks_path={MASKS_PATH} "
    "data.limit=100 "
    f"logger.experiment_name={EXPERIMENT_NAME} "
    "replacement=imagenet_mean "
    "batch_size=64 "
    "seed=42 "
    "logger.log_figures=true "
)

commands = [
    # GradCAM++ (single forward+backward pass, no budget parameter)
    (
        f"uv run python -m ciao.baselines.gradcam "
        f"{COMMON}"
        f"gradcam.variant=gradcampp "
        f"logger.run_name=gradcampp "
    ),
    # Occlusion Ã¢â‚¬â€ small window + tight stride for a smooth dense heatmap
    (
        f"uv run python -m ciao.baselines.occlusion "
        f"{COMMON}"
        f"occlusion.window_size=16 "
        f"occlusion.stride=2 "
        f"logger.run_name=occlusion "
    ),
    #    # Meaningful Perturbations Ã¢â‚¬â€ 5x longer than default (1500 iters / 5 min)
    #    (
    #        f"uv run python -m ciao.baselines.meaningful_perturbations "
    #        f"{COMMON}"
    #        f"mp.max_iterations=1500 "
    #        f"mp.max_time=300 "
    #        f"logger.run_name=mp "
    #    ),
    # LIME with SLIC segmentation Ã¢â‚¬â€ 3x more segments, 10x more samples
    (
        f"uv run python -m ciao.baselines.lime "
        f"{COMMON}"
        f"segmentation=slic "
        f"segmentation.n_segments=600 "
        f"lime.n_samples=10000 "
        f"logger.run_name=lime-slic "
    ),
]

submit_job(
    job_name="ciao-occlusion",
    username="dhalmazna",
    storage=[storage.secure.PROJECTS],
    public=False,
    cpu=1,
    memory="4Gi",
    gpu="H100",
    script=[
        "git clone https://github.com/RationAI/ciao.git",
        "cd ciao",
        f"git checkout {BRANCH}",
        "uv sync",
        "export MLFLOW_TRACKING_USERNAME=YOUR_MLFLOW_USERNAME",
        "export MLFLOW_TRACKING_PASSWORD=YOUR_MLFLOW_PASSWORD",
        *commands,
    ],
)
