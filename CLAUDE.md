# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### SAE Training
```bash
# Train SAE for baseline model
bash scripts/SW_MD_BM.sh

# Train SAE for RL model  
bash scripts/SW_MD_RL.sh

# Train SAE with SLURM (for cluster environments)
bash scripts/SW_MD_BM_slurm.sh
bash scripts/SW_MD_RL_slurm.sh

# Train 2B parameter SAE
bash scripts/train_sae_2b.sh
```

### Sequence Generation and Analysis
```bash
# Run protein sequence generation
bash scripts/run_generation.sh

# Predict protein activity using oracle models
bash scripts/run_activity_predictions.sh

# Fold proteins using ESMFold
bash scripts/run_folding.sh

# Score generated sequences
bash scripts/run_scoring.sh

# Create unified dataframes from analysis data
bash scripts/create_dataframe.sh
```

### Dataset Creation
```bash
# Create diffing datasets for SAE training
bash scripts/create_datasets_diffing.sh

# Create mixture datasets
bash scripts/create_datasets_diffing_mixture.sh
```

### Complete Pipeline
```bash
# Run the full analysis pipeline
bash scripts/pipeline.sh
```

## Code Architecture

### Core Training System (`src/training/`)
- **`training.py`**: Main SAE training logic with gradient accumulation, checkpointing, and threshold computation
- **`sae.py`**: Multiple SAE implementations (BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE)
- **`activation_store.py`**: Manages model activations, dataset loading, and batch generation
- **`config.py`**: Configuration management with hierarchical defaults
- **`logs.py`**: WandB integration, performance monitoring, and checkpointing

### Inference and Evaluation (`src/inference/`)
- **`sae_eval.py`**: SAE model evaluation utilities
- **`compute_threshold.py`**: Activation threshold computation
- **`inference_batch_topk.py`**: Batch inference for top-k SAEs

### Analysis Tools (`src/tools/`)
- **`diffing/`**: Latent space analysis, firing pattern analysis, and feature correlation
- **`generate/`**: Sequence generation with ablation, steering, and penalty mechanisms
- **`oracles/`**: Protein folding (ESMFold) and activity prediction using ESM models
- **`data_utils/`**: Dataset creation and manipulation utilities
- **`plots/`**: Visualization tools for analysis results

### Configuration System
- **YAML-based configuration** with hierarchical inheritance
- **Base config**: `configs/base_config.yaml` provides defaults
- **Experiment-specific configs**: Override base settings for specific experiments
- **Key config sections**: `base`, `sae`, `training`, `resuming`

### Experimental Framework
The codebase implements a pipeline for training and analyzing Sparse Autoencoders on protein language models:

1. **Sequence Generation**: Generate protein sequences using ZymCTRL for specific enzyme classes (e.g., alpha-amylases 3.2.1.1)
2. **Protein Folding**: Fold generated sequences using ESMFold
3. **Activity Prediction**: Predict enzymatic activity using oracle ESM models
4. **SAE Training**: Train sparse autoencoders on generated sequences to identify important features
5. **Feature Analysis**: Analyze feature evolution, correlations, and fitness across training iterations

### Key Data Paths
- **Generated sequences**: `/seq_gens/seq_gen_{ec_label}_iteration{N}.fasta`
- **Activity predictions**: `/activity_predictions/activity_prediction_iteration{N}.txt`
- **PDB structures**: `/outputs/output_iterations{N}/PDB/`
- **SAE checkpoints**: Generated automatically in training
- **Analysis data**: `/Diffing_Analysis_Data/`

### Multi-environment Support
- **Local development**: Use base configs and local scripts
- **SLURM clusters**: Use `*_slurm.sh` scripts with proper resource allocation
- **GPU requirements**: Most training requires A40/A100 GPUs

### Dependencies
- PyTorch ecosystem with CUDA support
- Transformers and SAE-lens for model implementations
- ESM models for protein analysis
- WandB for experiment tracking
- Various protein analysis tools (foldseek, etc.)