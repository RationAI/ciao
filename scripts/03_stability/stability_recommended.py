from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/grid_search/images"
EXPERIMENT_NAME = "mcgs-stability-recommended"
BRANCH = "test/all-algorithms"

CALIBRATED_EVALS = {
    64: 100000,
}

ALPHA_VALUES = [0.0]
C_VALUES = [1.4]

alpha_sweep = ",".join(str(a) for a in ALPHA_VALUES)
c_sweep = ",".join(str(c) for c in C_VALUES)


rollouts = 64
num_evals = CALIBRATED_EVALS[rollouts]
job_name = "mcgs-stability-recommended-90"

cmd = (
    f"uv run python -m ciao "
    f"logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
    f"data=directory_batch "
    f"data.batch_path={IMAGES_PATH} "
    f"data.limit=30 "
    f"logger.experiment_name={EXPERIMENT_NAME} "
    f"method=mcgs "
    f"method.num_evals={num_evals} "
    f"method.num_rollouts={rollouts} "
    f"method.exploration_c={c_sweep} "
    f"method.alpha={alpha_sweep} "
    f"segmentation=hexagonal "
    f"replacement=imagenet_mean "
    f"sigma=1 "
    f"desired_length=90 "  # we swept the length manually, it was 15, 30 and 90
    f"batch_size=64 "
    f"seed=1,2,3,4,5,6,7,8,9,10 "
    f"logger.run_name='mcgs-${{seed}}-length-90' "
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
