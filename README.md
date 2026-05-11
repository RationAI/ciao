# CIAO: Contextual Importance Assessment via Obfuscation

An implementation of explainable AI techniques for image classification. CIAO identifies influential image regions by systematically segmenting images, obfuscating segments, and using search algorithms to find important regions.

## Overview

CIAO explains what regions of an image contribute to a neural network's classification decisions. The method:

1. Segments the image into small regions
2. Obfuscates each segment and measures impact on model predictions
3. Uses search algorithms to group adjacent important segments into regions
4. Generates explanations showing which regions influenced the prediction

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/RationAI/ciao.git
cd ciao

# Install dependencies using uv
uv sync
```

### Basic Usage

Explain a single image with default settings:

```bash
uv run ciao
```

Customize the explanation using Hydra configuration overrides:

```bash
uv run ciao data.image_path=./my_image.jpg method=lookahead segmentation=square
```

Alternatively, run as a module:

```bash
uv run python -m ciao
```

### Development Commands

- `uv sync` - Install all dependencies
- `uv add <package>` - Add a new dependency
- `uv run ruff check` - Run linting
- `uv run ruff format` - Format code
- `uv run mypy .` - Run type checking
- `uv run ciao` - Run CIAO with default configuration
- `uv run pytest tests` - Execute tests

## Method Details

### How CIAO Works

1. **Segmentation**: The input image is divided into small regions (segments) using hexagonal or square grids
2. **Score Calculation**: Each segment is obfuscated (replaced) and the model is queried to measure how much that segment affects the prediction. This gives an importance score to each segment
3. **Region Search**: A search algorithm finds groups of adjacent segments with high importance scores, creating "regions" that represent influential image regions
4. **Explanation**: The top regions are visualized to show which regions most influenced the model's prediction

### Search Algorithms

- **UCB**: Upper Confidence Bound–guided sequential search
- **MCTS (Monte Carlo Tree Search)**: Tree-based search with UCB exploration
- **MC-RAVE**: MCTS with Rapid Action Value Estimation
- **MCGS (Monte Carlo Graph Search)**: Graph-based variant allowing revisiting of states
- **MCGS-RAVE**: MCGS with RAVE enhancements
- **Lookahead**: Greedy search with lookahead
- **Potential**: Potential field-guided sequential search
- **Pure Monte Carlo**: Basic Monte Carlo sampling
- **Beam Search**: Beam search over precomputed segment scores

### Segmentation Methods

- **Hexagonal Grid**: Divides image into hexagonal cells for better spatial coverage
- **Square Grid**: Simple square grid segmentation
- **SLIC**: Superpixel segmentation via Simple Linear Iterative Clustering

### Replacement Methods

- **Mean Color**: Replace masked regions with the image's mean color (normalized)
- **ImageNet Mean**: Replace masked regions with the dataset ImageNet mean color
- **Blur**: Gaussian blur applied to masked regions
- **Interlacing**: Interlaced pattern replacement
- **Solid Color**: Replace with a specified solid color (RGB)

### Baselines

- **GradCAM++**: Gradient-weighted class activation mapping
- **LIME**: Local Interpretable Model-agnostic Explanations (with SLIC superpixels)
- **Meaningful Perturbations**: Mask optimization-based explanations
- **Occlusion**: Sliding-window occlusion sensitivity
- **Extremal Perturbations**: Extremal perturbation-based saliency

## Project Structure

```
ciao/
├── ciao/                           # Main package
│   ├── algorithm/                  # Search algorithms and data structures
│   │   ├── builder.py              # Unified region builder orchestrating searches
│   │   ├── context.py              # Search context configurations
│   │   ├── graph.py                # Graph helpers
│   │   ├── lookahead.py            # Greedy lookahead
│   │   ├── mcgs.py                 # Monte Carlo Graph Search
│   │   ├── mcts.py                 # Monte Carlo Tree Search
│   │   ├── nodes.py                # Node classes for tree/graph search
│   │   ├── potential.py            # Potential-based search
│   │   ├── pure_monte_carlo.py     # Pure Monte Carlo sampling
│   │   ├── beam_search_precomputed.py  # Beam search over precomputed scores
│   │   ├── ucb.py                  # UCB-guided search
│   │   └── search_helpers.py       # Shared search helper functions
│   ├── baselines/                  # Baseline explainability methods
│   │   ├── extremal/               # Extremal perturbations
│   │   ├── gradcam/                # GradCAM++
│   │   ├── lime/                   # LIME
│   │   ├── meaningful_perturbations/  # Meaningful perturbations
│   │   └── occlusion/              # Occlusion sensitivity
│   ├── data/                       # Data loading and preprocessing
│   │   ├── constants.py            # Dataset constants (ImageNet mean/std, etc.)
│   │   ├── imagenet_s.py           # ImageNet-S dataset utilities
│   │   ├── loader.py               # Path loaders
│   │   ├── preprocessing.py        # Image preprocessing utilities
│   │   ├── replacement.py          # Image obfuscation / replacement strategies
│   │   └── segmentation.py         # Segmentation utilities (hex/square/SLIC)
│   ├── explainer/                  # Core explainer implementation
│   │   ├── ciao_explainer.py       # Main CIAO explainer class
│   │   └── explanation_methods.py  # Methods for the explanation algorithms
│   ├── metrics/                    # Evaluation metrics
│   │   ├── saliency.py             # Saliency map metrics
│   │   └── segmentation.py         # Segmentation quality metrics
│   ├── model/                      # Model inference and predictions
│   │   ├── classes.py              # Class label utilities
│   │   ├── pcam.py                 # PatchCamelyon model support
│   │   └── predictor.py            # ModelPredictor class for inference
│   ├── scoring/                    # Scoring
│   │   ├── segments.py             # Surrogate dataset creation and segment scoring
│   │   └── region.py               # Region evaluation and selection
│   ├── visualization/              # Visualization tools
│   │   └── visualization.py        # Saliency and region visualizations
│   ├── typing.py                   # Type aliases and definitions
│   └── __main__.py                 # CLI entry point
├── configs/                        # Hydra configuration files
│   ├── base.yaml                   # Base CIAO config (defaults + hyperparameters)
│   ├── ep_base.yaml                # Extremal perturbations base config
│   ├── gradcam_base.yaml           # GradCAM baseline base config
│   ├── lime_base.yaml              # LIME baseline base config
│   ├── mp_base.yaml                # Meaningful perturbations base config
│   ├── occlusion_base.yaml         # Occlusion baseline base config
│   ├── saliency_analysis.yaml      # Saliency analysis config
│   ├── baselines_saliency_analysis.yaml  # Baselines saliency analysis config
│   ├── classes/                    # Class name lists
│   │   ├── imagenet.yaml
│   │   └── pcam.yaml
│   ├── data/                       # Data source configurations
│   │   ├── single_image.yaml       # Single image via image_path
│   │   ├── directory_batch.yaml    # Directory batch with limit
│   │   └── imagenet_s_batch.yaml   # ImageNet-S batch config
│   ├── logger/                     # Experiment tracker settings
│   │   └── mlflow.yaml
│   ├── method/                     # Search algorithm configs
│   │   ├── beam_search.yaml
│   │   ├── lookahead.yaml
│   │   ├── mcgs.yaml
│   │   ├── mcts.yaml
│   │   ├── potential.yaml
│   │   ├── pure_monte_carlo.yaml
│   │   └── ucb.yaml
│   ├── model/                      # Model backbone configs
│   │   ├── resnet50.yaml
│   │   └── pcam_resnet50.yaml
│   ├── preprocessing/              # Preprocessing pipeline configs
│   │   ├── imagenet.yaml
│   │   └── pcam.yaml
│   ├── replacement/                # Obfuscation strategy configs
│   │   ├── blur.yaml
│   │   ├── imagenet_mean.yaml
│   │   ├── interlacing.yaml
│   │   ├── mean_color.yaml
│   │   ├── mean_color_pcam.yaml
│   │   └── solid_color.yaml
│   ├── runs/                       # Experiment run compositions
│   │   └── batch_example.yaml
│   └── segmentation/               # Segmentation strategy configs
│       ├── hexagonal.yaml
│       ├── slic.yaml
│       └── square.yaml
├── funnybirds_eval/                # FunnyBirds benchmark evaluation
│   ├── ciao_explainer.py           # CIAO explainer adapter for FunnyBirds
│   └── evaluate_explainability.py  # Evaluation entry point
├── notebooks/                      # Analysis notebooks
│   ├── 02_grid_search/             # Hyperparameter grid search analysis
│   ├── 03_stability/               # Algorithm stability analysis
│   ├── 04_internal_comparison/     # Internal algorithm comparison
│   ├── 05_sota_comparison_imagenet/  # SotA comparison on ImageNet-S
│   ├── 06_pcam/                    # PCam dataset experiments
│   └── multiple_regions/           # Multiple-region saliency analysis
├── scripts/                        # Experiment runner scripts
│   ├── 01_download_datasets/       # Dataset download helpers
│   ├── 02_grid_search/             # Grid search runs
│   ├── 03_stability/               # Stability experiment runs
│   ├── 04_internal_comparison/     # Internal comparison runs
│   ├── 05_sota_comparison_imagenet/  # ImageNet-S SotA comparison runs
│   ├── 06_sota_comparison_pcam/    # PCam SotA comparison runs
│   ├── 07_funnybirds/              # FunnyBirds evaluation runs
│   ├── multiple_regions/           # Multiple-region experiment runs
│   └── ucb_slic/                   # UCB + SLIC experiment runs
├── tests/                          # Test suite
├── thesis_figures/                 # Thesis figure generation notebooks
├── tools/                          # Standalone utility scripts
│   ├── compute_saliency.py         # Compute CIAO saliency maps
│   └── compute_baselines_saliency.py  # Compute baseline saliency maps
└── pyproject.toml                  # Project metadata and dependencies
```
