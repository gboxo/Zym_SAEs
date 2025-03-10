# Make src a package
from .utils import (
    load_model, 
    load_sae, 
    load_config, 
    get_ht_model, 
    convert_GPT_weights
)

from .training.config import (
    get_default_cfg, 
    update_cfg, 
    post_init_cfg
)

# Import SAE models
from .training.sae import (
    BaseAutoencoder,
    BatchTopKSAE,
    TopKSAE,
    VanillaSAE,
    JumpReLUSAE
)

# Import training utilities
from .training.training import train_sae, resume_training
from .training.logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
from .training.activation_store import ActivationsStore

# Import inference utilities
from .inference.inference_batch_topk import convert_to_jumprelu
from .inference.compute_threshold import compute_threshold
from .inference.sae_eval import SAEEval, EvalConfig 