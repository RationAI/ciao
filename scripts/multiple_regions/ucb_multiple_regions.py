from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/images"
MASKS_PATH = "CIAO_DATA_ROOT/imagenet_s/comparison/masks"
EXPERIMENT_NAME = "ucb-multiple-regions"
BRANCH = "test/all-algorithms"

# num_regions -> per-region desired_lengths
# Total selected segments match 1-region lengths [15, 30, 60, 90]:
#   2R: 2Ãƒâ€”[8,15,30,45]  -> totals [16, 30, 60, 90]
#   3R: 3Ãƒâ€”[5,10,20,30]  -> totals [15, 30, 60, 90]
#   4R: 4Ãƒâ€”[4, 8,15,23]  -> totals [16, 32, 60, 92]
REGIONS_CONFIG: dict[int, list[int]] = {
    2: [8, 15, 30, 45],
    3: [5, 10, 20, 30],
    4: [4, 8, 15, 23],
}

# Total model evals per image per run Ã¢â‚¬â€ matches the 1-region baseline
# (step_budget = round(NUM_EVALS / (num_regions * desired_length)))
NUM_EVALS = 100_000

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
for num_regions, lengths in REGIONS_CONFIG.items():
    for desired_length in lengths:
        # Total evals = step_budget * num_regions * desired_length Ã¢â€°Ë† NUM_EVALS
        step_budget = int(round(NUM_EVALS / (num_regions * desired_length)))

        method_commands.append(
            build_cmd(
                "ucb",
                (
                    f"max_regions={num_regions} "
                    f"method.step_budget={step_budget} "
                    f"method.batch_size={UCB_BATCH_SIZE} "
                    f"method.ucb_c={UCB_C} "
                    f"method.ucb_alpha={UCB_ALPHA}"
                ),
                f"ucb-{num_regions}regions",
                desired_length,
            )
        )

submit_job(
    job_name="ucb-multiple-regions",
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
