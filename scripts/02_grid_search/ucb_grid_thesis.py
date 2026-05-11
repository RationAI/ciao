from kube_jobs import storage, submit_job


IMAGES_PATH = "CIAO_DATA_ROOT/imagenet_s/grid_search/images"
EXPERIMENT_NAME = "ucb-grid-thesis"
BRANCH = "test/all-algorithms"

# UCB total evals Ã¢â€°Ë† step_budget Ãƒâ€” desired_length (30).
# step_budget is the UCB analogue of num_evals / desired_length.
# batch_size is the UCB analogue of num_rollouts (candidates scored per step).
#
# Calibrated to match the MCTS/MCGS wall-time tiers (~30s cap on H100):
#   batch_size=1,  step_budget=300  Ã¢â€ â€™ ~9 000 evals  (~8s)
#   batch_size=16, step_budget=2767 Ã¢â€ â€™ ~83 000 evals (~28s)
#   batch_size=64, step_budget=3333 Ã¢â€ â€™ ~100 000 evals (~30s)
CALIBRATED = {
    1: 300,
    16: 2767,
    64: 3333,
}

UCB_C_VALUES = [0.1, 0.7, 1.4, 3, 5]
UCB_ALPHA_VALUES = [0, 0.5, 1]

ucb_c_sweep = ",".join(str(c) for c in UCB_C_VALUES)
ucb_alpha_sweep = ",".join(str(a) for a in UCB_ALPHA_VALUES)

for batch_size, step_budget in CALIBRATED.items():
    job_name = f"ucb-grid-bs{batch_size}-sb{step_budget}"

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
        f"desired_length=30 "
        f"batch_size=64 "
        f"logger.run_name='ucb_bs_{batch_size}_sb_{step_budget}_c_${{method.ucb_c}}_a_${{method.ucb_alpha}}' "
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
