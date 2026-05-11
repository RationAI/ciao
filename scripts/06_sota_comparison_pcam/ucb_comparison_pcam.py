from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/pcam_sample"
EXPERIMENT_NAME = "ucb-comparison-pcam"
BRANCH = "test/all-algorithms"

DESIRED_LENGTHS = [10, 20, 40, 60]

UCB_EVALS_TOTAL = 100000
UCB_C = 1.4
UCB_ALPHA = 0.5

COMMON = (
    "logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
    "data=directory_batch "
    f"data.batch_path={IMAGES_PATH} "
    "data.limit=100 "
    f"logger.experiment_name={EXPERIMENT_NAME} "
    "model=pcam_resnet50 "
    "classes=pcam "
    "preprocessing=pcam "
    "segmentation=hexagonal "
    "replacement=blur "
    "sigma=1 "
    "batch_size=64 "
    "seed=42 "
)

ucb_commands = [
    (
        f"uv run python -m ciao "
        f"{COMMON} "
        f"method=ucb "
        f"method.step_budget={UCB_EVALS_TOTAL // length} "
        f"method.ucb_c={UCB_C} "
        f"method.ucb_alpha={UCB_ALPHA} "
        f"desired_length={length} "
        f"logger.run_name='ucb-42-length-{length}' "
        f"-m"
    )
    for length in DESIRED_LENGTHS
]

saliency_command = (
    "uv run python tools/compute_saliency.py "
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
    "masks_path=null "
    "iou_top_fraction=null "
    "sigma_fraction=0.03 "
    "log_figures=true "
)

submit_job(
    job_name="ucb-pcam",
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
        *ucb_commands,
        saliency_command,
    ],
)
