# Sparse Autoencoder Training System Documentation

## Overview
This system implements and trains sparse autoencoders (SAEs) for language models. It supports multiple SAE variants and includes comprehensive training utilities.

## Key Components

### 1. Training Module (`training.py`)
Core training logic for sparse autoencoders.

#### Main Functions:
- **`train_sae()`**: Main training loop
- **`resume_training()`**: Resume training from checkpoint
- **`generate_checkpoint_dir()`**: Creates standardized checkpoint directories
- **`threshold_loop_collect()`**: Collects activation thresholds
- **`threshold_loop_compute()`**: Computes activation thresholds

#### Key Features:
- Gradient accumulation
- Checkpointing
- Performance logging
- Threshold computation
- Multiple SAE variants support

### 2. Activation Store (`activation_store.py`)
Manages storage and retrieval of model activations.

#### Key Features:
- Dataset loading and caching
- Batch generation
- Position tracking
- Efficient skipping to specific positions
- Activation buffer management

### 3. Configuration (`config.py`)
Handles training configuration.

#### Key Functions:
- **`get_default_cfg()`**: Returns default configuration
- **`post_init_cfg()`**: Post-processes configuration
- **`update_cfg()`**: Updates configuration

### 4. Logging (`logs.py`)
Handles logging and checkpointing.

#### Key Features:
- WandB integration
- Model performance evaluation
- Checkpoint management
- Gradient statistics
- Feature activation tracking

### 5. SAE Implementations (`sae.py`)
Contains different SAE variants.

#### Implemented Variants:
1. **BaseAutoencoder**
   - Core autoencoder functionality
   - Weight normalization
   - Input preprocessing

2. **BatchTopKSAE**
   - Batch-wise top-k sparsity
   - Auxiliary loss for dead features

3. **TopKSAE**
   - Standard top-k sparsity
   - Similar to BatchTopK but without batch-wise operations

4. **VanillaSAE**
   - Basic sparse autoencoder
   - No top-k sparsity

5. **JumpReLUSAE**
   - Implements jump ReLU activation
   - Learns activation thresholds
   - Smooth gradient approximation

## Training Workflow

1. **Initialization**
   - Load configuration
   - Initialize model and SAE
   - Set up activation store

2. **Training Loop**
   - Forward pass through SAE
   - Loss computation
   - Backpropagation
   - Weight updates

3. **Monitoring**
   - Performance metrics
   - Threshold computation
   - Checkpointing

4. **Completion**
   - Final checkpoint
   - Cleanup
   - Logging final metrics

## Usage

### Starting New Training
```python
from src.training import train_sae

# Initialize model and config
model = ...
cfg = ...

# Start training
sae, checkpoint_dir = train_sae(model, cfg)
```

### Resuming Training
```python
from src.training import resume_training

# Resume training
sae, checkpoint_dir = resume_training(model, cfg, checkpoint_path)
```

## Configuration
The system uses a hierarchical configuration system with sensible defaults. Key configuration parameters include:

- Model architecture
- Training hyperparameters
- SAE dimensions
- Logging settings
- Checkpoint frequency
- Threshold computation parameters

## Checkpointing
The system provides comprehensive checkpointing capabilities:

- Automatic checkpoint directory generation
- Periodic checkpointing
- Final checkpoint
- Resume functionality
- Activation store state preservation

## Performance Monitoring
The system tracks multiple performance metrics:

- Reconstruction quality
- Feature activation statistics
- Gradient norms
- Dead feature tracking
- Model performance degradation

## Customization
The system is designed to be extensible:

- Add new SAE variants by subclassing BaseAutoencoder
- Modify training loop behavior
- Add new monitoring metrics
- Customize checkpointing behavior
