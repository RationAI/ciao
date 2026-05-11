"""Kubernetes job: evaluate CIAO on the FunnyBirds benchmark.

The FunnyBirds dataset (test split + interventions) is committed to the CIAO
repo under data/funnybirds/FunnyBirds/ and is available after git clone.

The VGG16 checkpoint is on cluster storage at MODEL_CHECKPOINT_PATH.
Set DOWNLOAD_MODEL = True only if it is not yet there.

To submit: uv run python jobs_testing/funnybirds/evaluate_funnybirds.py
"""

from kube_jobs import storage, submit_job


# Ã¢â€â‚¬Ã¢â€â‚¬ Cluster paths Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Dataset is on persistent cluster storage (transferred manually).
FUNNYBIRDS_DATA_PATH = "CIAO_DATA_ROOT/funnybirds/FunnyBirds"
MODEL_CHECKPOINT_PATH = (
    "CIAO_DATA_ROOT/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar"
)
# Results are written to persistent storage so they survive after the job ends.
MLFLOW_TRACKING_URI = "https://mlflow.rationai.cloud.e-infra.cz/"
MLFLOW_EXPERIMENT = "ciao-funnybirds"

# Ã¢â€â‚¬Ã¢â€â‚¬ Download toggles Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# Dataset: no download needed Ã¢â‚¬â€ it's in the repo.
# Model: already on cluster storage; set True only if the file is missing.
DOWNLOAD_MODEL = False

# Ã¢â€â‚¬Ã¢â€â‚¬ CIAO / evaluation config Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
BRANCH = "test/all-algorithms"

CIAO_METHOD = "ucb"
CIAO_SEGMENTATION = "slic"
CIAO_HEX_RADIUS = 4
CIAO_SLIC_N_SEGMENTS = 800
CIAO_SLIC_COMPACTNESS = 10.0
CIAO_MAX_REGIONS = 5
CIAO_DESIRED_LENGTH = 12  # ~30 segs per region ~ 1 bird part
CIAO_BATCH_SIZE = 64  # internal forward-pass batch size
CIAO_STEP_BUDGET = 500  # UCB/potential step budget

NR_ITRS = 500  # ~3 min/iter Ã¢â€ â€™ ~8h; set to 2501 for the full test set

_seg_tag = (
    f"r{CIAO_HEX_RADIUS}"
    if CIAO_SEGMENTATION == "hex"
    else f"n{CIAO_SLIC_N_SEGMENTS}"
    if CIAO_SEGMENTATION == "slic"
    else f"sq{CIAO_HEX_RADIUS}"
)
MLFLOW_RUN_NAME = (
    f"ciao-{CIAO_METHOD}-{CIAO_SEGMENTATION}-{_seg_tag}-reg{CIAO_MAX_REGIONS}"
)
RESULTS_PATH = f"CIAO_DATA_ROOT/funnybirds/results/ciao_{CIAO_METHOD}.txt"

# Ã¢â€â‚¬Ã¢â€â‚¬ Metrics to evaluate Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
# --accuracy works with test images only.
# All other metrics require test_interventions images (rendered separately).
# Comment out metrics whose intervention images are not yet committed.
METRICS = [
    "--accuracy",
    "--controlled_synthetic_data_check",
    "--single_deletion",
    "--preservation_check",
    "--deletion_check",
    "--target_sensitivity",
    "--distractibility",
    "--background_independence",
]


# Ã¢â€â‚¬Ã¢â€â‚¬ Build script Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
setup = [
    # Set up CIAO
    "git clone https://github.com/RationAI/ciao.git",
    "cd ciao",
    f"git checkout {BRANCH}",
    "uv sync --extra funnybirds",
    "export MLFLOW_TRACKING_USERNAME=YOUR_MLFLOW_USERNAME",
    "export MLFLOW_TRACKING_PASSWORD=YOUR_MLFLOW_PASSWORD",
    # Get the FunnyBirds framework (original) and inject our adapter
    "git clone https://github.com/visinf/funnybirds-framework.git",
    "cp funnybirds_eval/ciao_explainer.py funnybirds-framework/explainers/",
    "cp funnybirds_eval/evaluate_explainability.py funnybirds-framework/",
    # Fix Python 3.9+ incompatibility: base64.decodestring was removed
    "sed -i 's/from base64 import decodestring, decodebytes/from base64 import decodebytes/' funnybirds-framework/datasets/funny_birds.py",
    # Shuffle DataLoaders so nr_itrs samples are spread across all 50 classes
    "sed -i 's/shuffle=False/shuffle=True/g' funnybirds-framework/evaluation_protocols.py",
    # Fix CPU support: replace .cuda(args.gpu, ...) with device-agnostic .to(...)
    # "sed -i 's/\\.cuda(args\\.gpu,/.to(\"cpu\" if args.gpu < 0 else \"cuda:%d\" % args.gpu,/g' funnybirds-framework/evaluation_protocols.py",
]

if DOWNLOAD_MODEL:
    setup += [
        f"mkdir -p $(dirname {MODEL_CHECKPOINT_PATH})",
        f"wget -q https://download.visinf.tu-darmstadt.de/data/funnybirds/models/vgg16_final_1_checkpoint_best.pth.tar -O {MODEL_CHECKPOINT_PATH}",
    ]

setup += [f"mkdir -p $(dirname {RESULTS_PATH})"]

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
    job_name="ciao-funnybirds-eval-h",
    username="dhalmazna",
    storage=[storage.secure.PROJECTS],
    public=False,
    cpu=1,
    gpu="H100",
    memory="4Gi",
    script=[
        *setup,
        eval_cmd,
    ],
)
