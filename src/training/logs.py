import wandb
import torch
from torch.nn import functional as F
from functools import partial
import os
import json
import re
from collections import OrderedDict

def init_wandb(cfg, resume=False):
    """
    Initialize wandb with support for resuming.
    """
    if not cfg.training.use_wandb:
        return None
        
    import wandb
    
    # Set wandb directory if specified
    if cfg.training.wandb_dir:
        os.environ["WANDB_DIR"] = cfg.training.wandb_dir
    
    # Determine run name
    run_name = cfg.training.wandb_run_name
    if run_name is None:
        # Generate run name based on model and configuration
        model_name = os.path.basename(cfg.base.model_path.rstrip('/'))
        hook_name = cfg.training.hook_point.split('.')[-1]
        run_name = f"{model_name}_{hook_name}_{cfg.training.sae_type}_{cfg.training.top_k}_{cfg.training.lr}"
        
    if resume:
        run_name += "_resumed"
        
    # Resume wandb run if resuming and run_id is in config
    run_id = cfg.training.wandb_run_id if resume else None
    
    wandb_run = wandb.init(
        project=cfg.training.wandb_project,
        name=run_name,
        config=cfg,
        resume="must" if run_id else "never",
        id=run_id
    )
    
    # Store run ID in config for potential future resuming
    cfg.training.wandb_run_id = wandb_run.id
    
    return wandb_run

def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

# Hooks for model performance evaluation
def reconstr_hook(activation, hook, sae_out):
    return sae_out

def zero_abl_hook(activation, hook):
    return torch.zeros_like(activation)

def mean_abl_hook(activation, hook):
    return activation.mean([0, 1]).expand_as(activation)

def log_decoder_weights(sae, decoder_weights, step, wandb_run):
    actual_decoder_weights = sae.state_dict()["W_dec"].cpu()

    cosine_similarity = F.cosine_similarity(decoder_weights, actual_decoder_weights, dim=1)
    
    # Calculate statistics for cosine similarity
    mean_cosine = cosine_similarity.mean().item()
    min_cosine = cosine_similarity.min().item()
    max_cosine = cosine_similarity.max().item()
    std_cosine = cosine_similarity.std().item()
    
    # Log individual statistics instead of raw tensor to avoid histogram binning issues
    log_dict = {
        "performance/cosine_similarity_mean": mean_cosine,
        "performance/cosine_similarity_min": min_cosine,
        "performance/cosine_similarity_max": max_cosine,
        "performance/cosine_similarity_std": std_cosine
    }
    
    # Only log the full histogram data if there's enough variation
    if std_cosine > 1e-5 and max_cosine - min_cosine > 1e-5:
        log_dict["performance/cosine_similarity_hist"] = wandb.Histogram(cosine_similarity.detach().cpu().numpy())
    
    wandb_run.log(log_dict, step=step)
    



@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, index=None, batch_tokens=None):
    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()[:sae.cfg["batch_size"] // sae.cfg["seq_len"]]
    batch = activations_store.get_activations(batch_tokens).reshape(-1, sae.cfg["act_size"])

    sae_output = sae(batch)["sae_out"].reshape(batch_tokens.shape[0], batch_tokens.shape[1], -1)

    original_loss = model(batch_tokens, return_type="loss").item()
    reconstr_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], partial(reconstr_hook, sae_out=sae_output))],
        return_type="loss",
    ).item()
    zero_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], zero_abl_hook)],
        return_type="loss",
    ).item()
    mean_loss = model.run_with_hooks(
        batch_tokens,
        fwd_hooks=[(sae.cfg["hook_point"], mean_abl_hook)],
        return_type="loss",
    ).item()

    ce_degradation = original_loss - reconstr_loss
    zero_degradation = original_loss - zero_loss
    mean_degradation = original_loss - mean_loss

    log_dict = {
        "performance/ce_degradation": ce_degradation,
        "performance/recovery_from_zero": (reconstr_loss - zero_loss) / zero_degradation,
        "performance/recovery_from_mean": (reconstr_loss - mean_loss) / mean_degradation,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(sae, optimizer, cfg, iter_num, dir_path, activation_store=None, is_final=False):
    """
    Save checkpoint with all necessary information to resume training.
    
    Args:
        sae: The sparse autoencoder model
        optimizer: The optimizer
        cfg: Training configuration
        iter_num: Current iteration number
        dir_path: Directory to save the checkpoint
        activation_store: Activation store for dataset position (optional)
        is_final: Whether this is the final checkpoint
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    # Determine if this is a resumed training
    is_resumed = "resume_from" in cfg and cfg["resume_from"] is not None
    
    checkpoint = {
        'model_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'cfg': cfg,
        'iter_num': iter_num,
        'resume_history': cfg.get('resume_history', []),  # Track resume history
        'is_resumed': is_resumed,
    }
    if activation_store is not None:
        checkpoint['activation_store_state'] = {
            'current_batch_idx': activation_store.current_batch_idx,
            'current_epoch': activation_store.current_epoch,
            'dataset_position': activation_store.dataset_position
        }
    
    suffix = "_final" if is_final else ""
    checkpoint_path = os.path.join(dir_path, f"checkpoint_{iter_num}{suffix}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    # Save a copy as "latest" for easy resuming
    latest_path = os.path.join(dir_path, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, sae=None, optimizer=None, activation_store=None, device='cpu'):
    """
    Load checkpoint with all necessary information to resume training.
    
    Args:
        checkpoint_path: Path to the checkpoint file or directory
        sae: The sparse autoencoder model (optional)
        optimizer: The optimizer (optional)
        activation_store: Activation store (optional)
        device: Device to load the model to
        
    Returns:
        Tuple of (sae, optimizer, cfg, iter_num, activation_store_state)
    """


    # If checkpoint_path is a directory, look for latest checkpoint
    if os.path.isdir(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            checkpoint_path = latest_path
        else:
            # Find the checkpoint with the highest iteration number
            checkpoints = [f for f in os.listdir(checkpoint_path) if f.startswith("checkpoint_") and f.endswith(".pt")]
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
                
            # Extract iteration numbers and find the highest
            iter_nums = []
            for ckpt in checkpoints:
                match = re.search(r'checkpoint_(\d+)', ckpt)
                if match:
                    iter_nums.append(int(match.group(1)))
            
            if not iter_nums:
                raise ValueError(f"No valid checkpoints found in {checkpoint_path}")
                
            max_iter = max(iter_nums)
            checkpoint_path = os.path.join(checkpoint_path, f"checkpoint_{max_iter}.pt")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, OrderedDict) and len(list(checkpoint.keys())) == 4:
        checkpoint_dict = {}
        checkpoint_dict['model_state_dict'] = checkpoint
        checkpoint = checkpoint_dict
    
    # Try to extract config and other metadata
    if "cfg" in checkpoint:
        cfg = checkpoint['cfg']
    else:
        with open(os.path.join(os.path.dirname(checkpoint_path), "config.json"), "r") as f:
            cfg = json.load(f)

    if "iter_num" in checkpoint:
        iter_num = checkpoint['iter_num']
    else:
        iter_num = 0
    if "model_type" not in cfg:
        cfg["model_type"] = "BatchTopK"
    
    # Update resume history
    if 'resume_history' not in checkpoint:
        checkpoint['resume_history'] = []
    
    # Add current checkpoint to resume history if not already there
    if checkpoint_path not in checkpoint['resume_history']:
        checkpoint['resume_history'].append(checkpoint_path)
    
    # Update the config with resume history
    cfg["resume_history"] = checkpoint['resume_history']
    cfg["resume_from"] = checkpoint_path
    
    # Load model state if provided
    if sae is not None and 'model_state_dict' in checkpoint:
        sae.load_state_dict(checkpoint['model_state_dict'])
        
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Extract activation store state
    activation_store_state = checkpoint.get('activation_store_state', None)
    
    # Update activation store if provided
    if activation_store is not None and activation_store_state is not None:
        activation_store.current_batch_idx = activation_store_state['current_batch_idx']
        activation_store.current_epoch = activation_store_state['current_epoch']
        activation_store.dataset_position = activation_store_state['dataset_position']
    
    print(f"Loaded checkpoint from {checkpoint_path} (iteration {iter_num})")
    
    return sae, optimizer, cfg, iter_num, activation_store_state

