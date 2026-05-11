from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/images"
MASKS_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/masks"
EXPERIMENT_NAME = "ucb-multiple-regions"
BRANCH = "test/all-algorithms"

submit_job(
    job_name="compute-saliency-multiple-regions",
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
        (
            "uv run python tools/compute_saliency.py "
            f"images_path={IMAGES_PATH} "
            f"masks_path={MASKS_PATH} "
            f"experiment_name={EXPERIMENT_NAME} "
            "compute_deletion=true "
            "compute_insertion=true "
            "deletion_steps=64 "
            "insertion_steps=64 "
            "iou_top_fraction=0.0707 "
            "sigma_fraction=0.03 "
            "log_figures=true "
        ),
    ],
)
