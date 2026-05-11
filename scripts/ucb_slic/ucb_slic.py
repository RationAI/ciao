from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/images"
MASKS_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/masks"
EXPERIMENT_NAME = "ucb-slic"
BRANCH = "test/all-algorithms"

DESIRED_LENGTHS = [8, 15, 30, 45]

MCGS_NUM_EVALS = 100000

UCB_BATCH_SIZE = 64
UCB_C = 1.4
UCB_ALPHA = 0.5

SEED_SWEEP = "42"

COMMON = (
    "logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
    "data=imagenet_s_batch "
    f"data.batch_path={IMAGES_PATH} "
    f"data.masks_path={MASKS_PATH} "
    "data.limit=100 "
    f"logger.experiment_name={EXPERIMENT_NAME} "
    "segmentation=slic "
    "segmentation.n_segments=600 "
    "replacement=imagenet_mean "
    "sigma=1 "
    "batch_size=64 "
    f"seed={SEED_SWEEP} "
)


def build_cmd(method: str, overrides: str, run_label: str, desired_length: int) -> str:
    run_name = f"{run_label}-${{seed}}-length-{desired_length}"
    return (
        f"uv run python -m ciao "
        f"{COMMON} "
        f"method={method} "
        f"{overrides} "
        f"desired_length={desired_length} "
        f"logger.run_name='{run_name}' "
        f"-m"
    )


method_commands = []
for desired_length in DESIRED_LENGTHS:
    step_budget = int(round(MCGS_NUM_EVALS / desired_length))

    method_commands.extend(
        [
            build_cmd(
                "ucb",
                (
                    f"method.step_budget={step_budget} "
                    f"method.batch_size={UCB_BATCH_SIZE} "
                    f"method.ucb_c={UCB_C} "
                    f"method.ucb_alpha={UCB_ALPHA}"
                ),
                "ucb",
                desired_length,
            ),
        ]
    )

submit_job(
    job_name="ucb-slic",
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
        *method_commands,
    ],
)
