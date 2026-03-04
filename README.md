# CIAO: Contextual Importance Assessment via Obfuscation

An implementation of explainable AI techniques for image classification. CIAO identifies influential image regions by systematically segmenting images, obfuscating segments, and using search algorithms to find important regions (hyperpixels).

## Overview

CIAO explains what regions of an image contribute to a neural network's classification decisions. The method:

1. Segments the image into small regions
2. Obfuscates each segment and measures impact on model predictions
3. Uses search algorithms to group adjacent important segments into hyperpixels
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
uv run ciao data.image_path=./my_image.jpg explanation.method=mcts explanation.segment_size=8
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
3. **Hyperpixel Search**: A search algorithm finds groups of adjacent segments with high importance scores, creating "hyperpixels" that represent influential image regions
4. **Explanation**: The top hyperpixels are visualized to show which regions most influenced the model's prediction

### Search Algorithms

- **MCTS (Monte Carlo Tree Search)**: Tree-based search with UCB exploration
- **MC-RAVE**: MCTS with Rapid Action Value Estimation
- **MCGS (Monte Carlo Graph Search)**: Graph-based variant allowing revisiting of states
- **MCGS-RAVE**: MCGS with RAVE enhancements
- **Lookahead**: Greedy search with lookahead using efficient bitset operations
- **Potential**: Potential field-guided sequential search

### Segmentation Methods

- **Hexagonal Grid**: Divides image into hexagonal cells for better spatial coverage
- **Square Grid**: Simple square grid segmentation

### Replacement Methods

- **Mean Color**: Replace masked regions with the image's mean color (normalized)
- **Blur**: Gaussian blur applied to masked regions
- **Interlacing**: Interlaced pattern replacement
- **Solid Color**: Replace with a specified solid color (RGB)

## Proposed project Structure

```
ciao/
├── ciao/                           # Main package
│   ├── algorithm/                  # Search algorithms
│   │   ├── mcts.py                 # Monte Carlo Tree Search
│   │   ├── mcgs.py                 # Monte Carlo Graph Search
│   │   ├── lookahead_bitset.py     # Greedy lookahead with bitsets
│   │   └── potential.py            # Potential-based search
│   ├── data/                       # Data loading and preprocessing
│   │   ├── loader.py               # Image loaders
│   │   └── preprocessing.py        # Image preprocessing utilities
│   ├── explainer/                  # Core explainer implementation
│   │   └── ciao_explainer.py       # Main CIAO explainer class
│   ├── structures/                 # Data structures
│   │   ├── bitmask_graph.py        # Bitset operations for hyperpixels
│   │   └── nodes.py                # Node classes for tree/graph search
│   ├── utils/                      # Utility functions
│   │   ├── calculations.py         # Score calculations and predictions
│   │   ├── segmentation.py         # Segmentation utilities
│   │   └── search_utils.py         # Search algorithm utilities
│   ├── visualization/              # Visualization tools
│   │   └── visualization.py        # Interactive visualizations
│   └── __main__.py                 # CLI entry point
├── configs/                        # Hydra configuration files
│   ├── ciao.yaml                   # Main entry point
│   ├── base.yaml                   # Base configuration
│   ├── data/                       # Data configurations
│   │   └── default.yaml
│   ├── explanation/                # Explanation method configs
│   │   └── ciao_default.yaml       # Default CIAO parameters
│   ├── hydra/                      # Hydra settings
│   └── logger/                     # Logger configurations
└── pyproject.toml                  # Project metadata and dependencies
```
