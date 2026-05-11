from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/pcam_sample"
EXPERIMENT_NAME = "baselines-comparison-pcam"
BRANCH = "test/all-algorithms"

COMMON = (
    "logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
    "data=directory_batch "
    f"data.batch_path={IMAGES_PATH} "
    "data.limit=100 "
    f"logger.experiment_name={EXPERIMENT_NAME} "
    "model=pcam_resnet50 "
    "classes=pcam "
    "preprocessing=pcam "
    "replacement=blur "
    "batch_size=64 "
    "seed=42 "
    "logger.log_figures=true "
)

commands = [
    # GradCAM++
    (
        f"uv run python -m ciao.baselines.gradcam "
        f"{COMMON}"
        f"gradcam.variant=gradcampp "
        f"logger.run_name=gradcampp "
    ),
    # Occlusion Ã¢â‚¬â€ window scaled for 96x96 (was 16/2 for 224x224)
    (
        f"uv run python -m ciao.baselines.occlusion "
        f"{COMMON}"
        f"occlusion.window_size=8 "
        f"occlusion.stride=1 "
        f"logger.run_name=occlusion "
    ),
    #    # Meaningful Perturbations
    #    (
    #        f"uv run python -m ciao.baselines.meaningful_perturbations "
    #        f"{COMMON}"
    #        f"mp.max_iterations=1500 "
    #        f"mp.max_time=300 "
    #        f"logger.run_name=mp "
    #    ),
    # LIME with SLIC Ã¢â‚¬â€ fewer segments for 96x96 (was 600 for 224x224)
    (
        f"uv run python -m ciao.baselines.lime "
        f"{COMMON}"
        f"segmentation=slic "
        f"segmentation.n_segments=100 "
        f"lime.n_samples=5000 "
        f"logger.run_name=lime-slic "
    ),
]

saliency_command = (
    "uv run python tools/compute_baselines_saliency.py "
    f"images_path={IMAGES_PATH} "
    f"experiment_name={EXPERIMENT_NAME} "
    "model=pcam_resnet50 "
    "classes=pcam "
    "preprocessing=pcam "
    "replacement=blur "
    "compute_deletion=true "
    "compute_insertion=true "
    "deletion_steps=64 "
    "insertion_steps=64 "
    "log_figures=true "
)

submit_job(
    job_name="baselines-pcam",
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
        saliency_command,
    ],
)
