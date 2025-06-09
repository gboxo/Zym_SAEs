# CRG Boxo: Sparse Autoencoder Analysis for Protein Sequences

## Overview

This repository contains code for training and analyzing Sparse Autoencoders (SAEs) on protein language models, with a focus on understanding feature representations and their relationship to protein activity.

The organic workflow for this repository is to use the `AI4PD/ZymCTRL` and the released SAEs `AI4PD/ZymCTRL-SAEs` along with a DMS datasource. For rough guidelines on how to change the pLM, SAE, or datasource, see the [Advanced Use](#advanced-use) section.


### Workflow

1) Downlaod the Deep Mutational Scan of choice.
2) Use the DMS to create an oracle that maps from seq-to-activity
3) Use a protein Language Model to sample enzymes of a given family (this could include approaches like fine-tuning, RL Alignment or prompting)
4) Score the samples with the oracle
5) Recover the SAE latents that are more predictive of high-low activity
6) Perform generation-time interventions using those latents
7) Score the generated variants

```{mermaid}
flowchart LR
    A["📥 Download Deep<br/>Mutational Scan (DMS)"] --> B["🎯 Create Oracle<br/>seq → activity mapping"]
    
    B --> C["🧬 Sample Protein Sequences<br/>using Language Model<br/>(Fine-tuning/RL/Prompting)"]
    
    C --> D["📊 Score Samples<br/>with Oracle"]
    
    style A fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    style B fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    style C fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    style D fill:#fff3e0,stroke:#e65100,stroke-width:2px
```

```{mermaid}
flowchart LR
    E["🔍 Identify Predictive<br/>SAE Latents<br/>(high vs low activity)"] --> F["⚡ Generation-time<br/>Interventions<br/>(ablation/clamping/steering)"]
    
    F --> G["✅ Score Generated<br/>Variants"]
    
    G --> H["🎉 Optimized Protein<br/>Variants"]
    
    style E fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    style F fill:#f1f8e9,stroke:#33691e,stroke-width:2px
    style G fill:#fff8e1,stroke:#ff6f00,stroke-width:2px
    style H fill:#e0f2f1,stroke:#00695c,stroke-width:3px
```





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


### 1. Setup

- Downlaod ZymCTRL
- Downlaod the SAE
- Downlaod the oracle
- Download the DMS data



### 2. Analyse the featuers

[Instructions for extracting and analyzing features]


**Scripts**

- `experiments_release/get_features_bash.sh`
- `experiments_release/latent_scoring_bash.sh`


### 3. Activity Prediction

[Instructions for running activity prediction]

**Scripts**

- `experiments_release/activity_prediction_bash.sh`

### 4. Sequence Generation

[Instructions for generating sequences with interventions]

- `experiments_release/generate_with_ablation_bash.sh`
- `experiments_release/generate_with_clammping_bash.sh`

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


## Data Format

[Description of expected data formats]


## Advanced Use

## Citation

[Citation information]

## License

[License information]

## Contributing

[Contributing guidelines]

## Contact

[Contact information]
