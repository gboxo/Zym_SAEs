# CRG Boxo: Sparse Autoencoder Analysis for Protein Sequences

## Overview

This repository contains code for training and analyzing Sparse Autoencoders (SAEs) on protein language models, with a focus on understanding feature representations and their relationship to protein activity.

## Features

- **SAE Training**: Train sparse autoencoders on protein language model representations
- **Feature Analysis**: Extract and analyze important features from trained SAEs
- **Activity Prediction**: Predict protein activity using oracle models
- **Sequence Generation**: Generate protein sequences with various intervention techniques
- **Interpretability**: Tools for understanding SAE feature representations

## Installation

### Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- See `requirements.txt` for complete dependency list

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd crg_boxo

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── training/           # SAE training components
│   ├── inference/          # Model inference and evaluation
│   ├── tools/              # Analysis and utility tools
│   │   ├── data_utils/     # Data processing utilities
│   │   ├── diffing/        # Feature extraction and analysis
│   │   ├── generate/       # Sequence generation tools
│   │   ├── oracles/        # Activity prediction models
│   │   └── kl_divergence/  # KL divergence analysis
│   └── config/             # Configuration loading
├── configs/                # Configuration files
├── experiments_release/    # Experiment scripts
└── config/                 # Main experiment configuration
```

## Quick Start

### 1. Training SAEs

[Instructions for training sparse autoencoders]

### 2. Feature Analysis

[Instructions for extracting and analyzing features]

### 3. Activity Prediction

[Instructions for running activity prediction]

### 4. Sequence Generation

[Instructions for generating sequences with interventions]

## Configuration

The project uses YAML configuration files to manage experiment parameters. Key configuration files:

- `config/experiment_config.yaml`: Main experiment configuration
- `configs/config_*.yaml`: Specific task configurations

## Experiments

### Released Experiments

The `experiments_release/` directory contains ready-to-run experiment scripts:

- `activity_prediction_bash.sh`: Run activity prediction on sequences
- `finetune.sh`: Fine-tune models
- `generate_sequences_bash.sh`: Generate protein sequences
- `generate_with_ablation_bash.sh`: Generate with feature ablation
- `generate_with_steering_bash.sh`: Generate with feature steering
- `latent_scoring_bash.sh`: Score latent features

### Running Experiments

[Instructions for running experiments]

## Key Components

### SAE Training (`src/training/`)

- **`sae.py`**: Core SAE implementation
- **`training.py`**: Training loop and optimization
- **`activation_store.py`**: Activation data management
- **`config.py`**: Training configuration

### Inference (`src/inference/`)

- **`sae_eval.py`**: SAE evaluation metrics
- **`inference_batch_topk.py`**: Batch inference with top-k features

### Analysis Tools (`src/tools/`)

#### Feature Analysis (`diffing/`)
- Feature extraction and importance scoring
- Latent space analysis

#### Sequence Generation (`generate/`)
- Controlled sequence generation
- Ablation and steering experiments

#### Activity Prediction (`oracles/`)
- Protein activity prediction using fine-tuned models
- Support for multiple oracle architectures

## Data Format

[Description of expected data formats]

## Results

[Information about expected outputs and results]

## Citation

[Citation information]

## License

[License information]

## Contributing

[Contributing guidelines]

## Contact

[Contact information]
