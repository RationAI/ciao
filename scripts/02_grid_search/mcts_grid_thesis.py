from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/grid_search/images"
EXPERIMENT_NAME = "mcts-grid-thesis"
BRANCH = "test/all-algorithms"

# Time-fair calibration: (num_rollouts -> num_evals) for approx. equal wall time per image.
# Derived from timing experiment (c=10, 30 images, MCGS) where each rollout count was measured
# to produce ~99686 evals/30s for rollouts=64. Lower rollout counts are slower per eval.
CALIBRATED_EVALS = {
    64: 100000,
    16: 83000,
    1: 9000,
}

ALPHA_VALUES = [0, 0.5, 1]
C_VALUES = [0.1, 0.7, 1.4, 3, 5]
ROLLOUT_VALUES = [1, 16, 64]

alpha_sweep = ",".join(str(a) for a in ALPHA_VALUES)
c_sweep = ",".join(str(c) for c in C_VALUES)

for rollouts in ROLLOUT_VALUES:
    num_evals = CALIBRATED_EVALS[rollouts]
    job_name = f"mcts-grid-r{rollouts}-e{num_evals}"

    cmd = (
        f"uv run python -m ciao "
        f"logger.tracking_uri=https://mlflow.rationai.cloud.e-infra.cz/ "
        f"data=directory_batch "
        f"data.batch_path={IMAGES_PATH} "
        f"data.limit=30 "
        f"logger.experiment_name={EXPERIMENT_NAME} "
        f"method=mcts "
        f"method.num_evals={num_evals} "
        f"method.num_rollouts={rollouts} "
        f"method.exploration_c={c_sweep} "
        f"method.alpha={alpha_sweep} "
        f"segmentation=hexagonal "
        f"replacement=imagenet_mean "
        f"sigma=1 "
        f"desired_length=30 "
        f"batch_size=64 "
        f"logger.run_name='mcts_r_{rollouts}_e_{num_evals}_c_${{method.exploration_c}}_a_${{method.alpha}}' "
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
