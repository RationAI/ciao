"""Kubernetes job: run CIAO on a small FunnyBirds sample for thesis visualizations.

Logs per-sample artifacts (original image, part map, attribution numpy arrays,
and publication-ready heatmap figures) to MLflow as nested runs.

To submit: uv run python scripts/07_funnybirds/visualize_funnybirds.py
"""

from kube_jobs import storage, submit_job


# Ã¢â€â‚¬Ã¢â€â‚¬ Cluster paths Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
FUNNYBIRDS_DATA_PATH = "CIAO_DATA_ROOT/funnybirds/FunnyBirds"
MODEL_CHECKPOINT_PATH = (
    "CIAO_DATA_ROOT/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar"
)
MLFLOW_TRACKING_URI = "https://mlflow.rationai.cloud.e-infra.cz/"
MLFLOW_EXPERIMENT = "ciao-funnybirds"

# Ã¢â€â‚¬Ã¢â€â‚¬ CIAO config (match the main evaluation run) Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
BRANCH = "test/all-algorithms"

CIAO_METHOD = "ucb"
CIAO_SEGMENTATION = "slic"
CIAO_HEX_RADIUS = 4
CIAO_SLIC_N_SEGMENTS = 800
CIAO_SLIC_COMPACTNESS = 10.0
CIAO_MAX_REGIONS = 5
CIAO_DESIRED_LENGTH = 12
CIAO_BATCH_SIZE = 64
CIAO_STEP_BUDGET = 500

# Small sample: enough to pick a few good examples for the thesis.
NR_ITRS = 20

_seg_tag = (
    f"r{CIAO_HEX_RADIUS}"
    if CIAO_SEGMENTATION == "hex"
    else f"n{CIAO_SLIC_N_SEGMENTS}"
    if CIAO_SEGMENTATION == "slic"
    else f"sq{CIAO_HEX_RADIUS}"
)
MLFLOW_RUN_NAME = (
    f"ciao-viz-{CIAO_METHOD}-{CIAO_SEGMENTATION}-{_seg_tag}-reg{CIAO_MAX_REGIONS}"
)
RESULTS_PATH = f"CIAO_DATA_ROOT/funnybirds/results/ciao_viz_{CIAO_METHOD}.txt"

# Only run CSDC (exercises get_part_importance Ã¢â€ â€™ triggers artifact logging with part maps).
METRICS = [
    "--controlled_synthetic_data_check",
]


# Ã¢â€â‚¬Ã¢â€â‚¬ Build script Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
setup = [
    "git clone https://github.com/RationAI/ciao.git",
    "cd ciao",
    f"git checkout {BRANCH}",
    "uv sync --extra funnybirds",
    "export MLFLOW_TRACKING_USERNAME=YOUR_MLFLOW_USERNAME",
    "export MLFLOW_TRACKING_PASSWORD=YOUR_MLFLOW_PASSWORD",
    "git clone https://github.com/visinf/funnybirds-framework.git",
    "cp funnybirds_eval/ciao_explainer.py funnybirds-framework/explainers/",
    "cp funnybirds_eval/evaluate_explainability.py funnybirds-framework/",
    "sed -i 's/from base64 import decodestring, decodebytes/from base64 import decodebytes/' funnybirds-framework/datasets/funny_birds.py",
    "sed -i 's/shuffle=False/shuffle=True/g' funnybirds-framework/evaluation_protocols.py",
    f"mkdir -p $(dirname {RESULTS_PATH})",
]

eval_cmd = (
    f"PYTHONPATH=funnybirds-framework "
    f"uv run python funnybirds-framework/evaluate_explainability.py "
    f"--data {FUNNYBIRDS_DATA_PATH} "
    f"--model vgg16 "
    f"--checkpoint_name {MODEL_CHECKPOINT_PATH} "
    f"--explainer CIAO "
    f"--ciao_method {CIAO_METHOD} "
    f"--ciao_segmentation {CIAO_SEGMENTATION} "
    f"--ciao_hex_radius {CIAO_HEX_RADIUS} "
    f"--ciao_slic_n_segments {CIAO_SLIC_N_SEGMENTS} "
    f"--ciao_slic_compactness {CIAO_SLIC_COMPACTNESS} "
    f"--ciao_max_regions {CIAO_MAX_REGIONS} "
    f"--ciao_desired_length {CIAO_DESIRED_LENGTH} "
    f"--ciao_batch_size {CIAO_BATCH_SIZE} "
    f"--ciao_step_budget {CIAO_STEP_BUDGET} "
    f"--nr_itrs {NR_ITRS} "
    + " ".join(METRICS)
    + f" --ciao_mlflow_tracking_uri {MLFLOW_TRACKING_URI}"
    + f" --ciao_mlflow_experiment {MLFLOW_EXPERIMENT}"
    + f" --ciao_mlflow_run_name {MLFLOW_RUN_NAME}"
    + f" --gpu -0 2>&1 | tee {RESULTS_PATH}"
)

submit_job(
    job_name="ciao-funnybirds-viz",
    username="dhalmazna",
    storage=[storage.secure.PROJECTS],
    public=False,
    cpu=1,
    gpu="A40",
    memory="4Gi",
    script=[
        *setup,
        eval_cmd,
    ],
)
