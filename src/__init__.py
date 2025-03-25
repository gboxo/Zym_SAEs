"""Sparse Autoencoder (SAE) package for language model interpretability.

This package provides tools for training and evaluating sparse autoencoders on 
language model activations. The main components are:

- SAE models: BaseAutoencoder, BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE
- Training utilities: train_sae, resume_training, ActivationsStore
- Inference utilities: SAEEval, compute_threshold, convert_to_jumprelu
- Configuration: get_default_cfg, update_cfg, post_init_cfg
- Model loading: load_model, load_sae, load_config, get_ht_model
"""

from .utils import (
    load_model,  # Load pretrained language models
    load_sae,    # Load trained sparse autoencoder
    load_config, # Load configuration files 
    get_ht_model, # Convert model to HookedTransformer format
    convert_GPT_weights # Convert GPT weights to SAE-compatible format
)

from .training.config import (
    get_default_cfg, 
    update_cfg, 
    post_init_cfg
)

# Import SAE models
from .training.sae import (
    BaseAutoencoder,  # Base class for all autoencoder models
    BatchTopKSAE,     # Sparse autoencoder with batch-wise top-k activation
    TopKSAE,          # Sparse autoencoder with global top-k activation
    VanillaSAE,       # Standard sparse autoencoder with L1 regularization
    JumpReLUSAE       # Sparse autoencoder with jump ReLU activation function
)

# Import training utilities
from .training.training import (
    train_sae,        # Main training loop for sparse autoencoders
    resume_training   # Resume training from checkpoint
)
from .training.logs import (
    init_wandb,       # Initialize Weights & Biases logging
    log_wandb,        # Log metrics to Weights & Biases
    log_model_performance, # Log model performance metrics
    save_checkpoint   # Save model checkpoint
)
from .training.activation_store import (
    ActivationsStore  # Store and manage model activations for training
)

# Import inference utilities
from .inference.inference_batch_topk import (
    convert_to_jumprelu  # Convert BatchTopKSAE to JumpReLUSAE format
)
from .inference.compute_threshold import (
    compute_threshold    # Compute activation thresholds for evaluation
)
from .inference.sae_eval import (
    SAEEval,   # Class for evaluating sparse autoencoder performance
    EvalConfig # Configuration for SAE evaluation
)
