from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/images"
MASKS_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/masks"
EXPERIMENT_NAME = "algorithm-comparison-imagenet-s"
BRANCH = "test/all-algorithms"

DESIRED_LENGTHS = [15, 30, 60, 90]
LOOKAHEAD_DISTANCE_BY_LENGTH = {15: 4, 30: 3, 60: 2, 90: 2}

MCGS_NUM_EVALS = 100000
MCGS_NUM_ROLLOUTS = 64
MCGS_C = 1.4
MCGS_ALPHA = 0.0

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
    "segmentation=hexagonal "
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
    lookahead_distance = LOOKAHEAD_DISTANCE_BY_LENGTH[desired_length]

    method_commands.extend(
        [
            build_cmd(
                "lookahead",
                f"method.lookahead_distance={lookahead_distance}",
                f"lookahead-d{lookahead_distance}",
                desired_length,
            ),
            build_cmd(
                "lookahead",
                "method.lookahead_distance=1",
                "greedy",
                desired_length,
            ),
            build_cmd(
                "beam_search",
                f"method.beam_width={step_budget}",
                f"beam-w{step_budget}",
                desired_length,
            ),
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
            build_cmd(
                "pure_monte_carlo",
                f"method.num_evals={MCGS_NUM_EVALS}",
                "pure-mc",
                desired_length,
            ),
            build_cmd(
                "potential",
                f"method.step_budget={step_budget}",
                "potential",
                desired_length,
            ),
            build_cmd(
                "mcgs",
                (
                    f"method.num_evals={MCGS_NUM_EVALS} "
                    f"method.num_rollouts={MCGS_NUM_ROLLOUTS} "
                    f"method.exploration_c={MCGS_C} "
                    f"method.alpha={MCGS_ALPHA}"
                ),
                "mcgs",
                desired_length,
            ),
        ]
    )

submit_job(
    job_name="algorithm-comparison-90",
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
