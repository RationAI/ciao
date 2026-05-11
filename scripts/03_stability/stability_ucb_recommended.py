from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/grid_search/images"
EXPERIMENT_NAME = "ucb-stability-recommended"
BRANCH = "test/all-algorithms"

# UCB step_budget calibration is defined for desired_length=30 in the grid search.
CALIBRATED_STEP_BUDGETS = {
    64: 3333,
}
BASE_DESIRED_LENGTH = 30
DESIRED_LENGTHS = [15, 30, 90]

UCB_C_VALUES = [1.4]
UCB_ALPHA_VALUES = [0.5]

ucb_c_sweep = ",".join(str(c) for c in UCB_C_VALUES)
ucb_alpha_sweep = ",".join(str(a) for a in UCB_ALPHA_VALUES)

batch_size = 64
for desired_length in DESIRED_LENGTHS:
    step_budget = (
        CALIBRATED_STEP_BUDGETS[batch_size] * BASE_DESIRED_LENGTH // desired_length
    )
    job_name = f"ucb-stability-recommended-{desired_length}"

    cmd = (
        f"uv run python -m ciao "
        f"logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
        f"data=directory_batch "
        f"data.batch_path={IMAGES_PATH} "
        f"data.limit=30 "
        f"logger.experiment_name={EXPERIMENT_NAME} "
        f"method=ucb "
        f"method.step_budget={step_budget} "
        f"method.batch_size={batch_size} "
        f"method.ucb_c={ucb_c_sweep} "
        f"method.ucb_alpha={ucb_alpha_sweep} "
        f"segmentation=hexagonal "
        f"replacement=imagenet_mean "
        f"sigma=1 "
        f"desired_length={desired_length} "
        f"batch_size=64 "
        f"seed=1,2,3,4,5,6,7,8,9,10 "
        f"logger.run_name='ucb-${{seed}}-length-{desired_length}' "
        f"-m"
    )

    submit_job(
        job_name=job_name,
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
            cmd,
        ],
    )
