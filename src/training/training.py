import torch
import tqdm
from .logs import init_wandb, log_wandb, log_model_performance, save_checkpoint, load_checkpoint, log_decoder_weights
import os
import re
from datetime import datetime
from ..config.paths import resolve_paths
import yaml
from .activation_store import ActivationsStore
from ..training.sae import BatchTopKSAE, TopKSAE, VanillaSAE, JumpReLUSAE, BaseAutoencoder

def generate_checkpoint_dir(cfg, resume=False):
    """
    Generate a checkpoint directory name that clearly indicates original properties
    and continuation without being too long.
    """
    # Extract base model name
    model_name = os.path.basename(cfg["model_name"].rstrip('_'))
    
    # Format date
    date_str = datetime.now().strftime("%d_%m_%y")
    
    # Base components of the name
    components = [
        model_name,
        date_str,
        f"{cfg["hook_point"].split('.')[-1]}",
        f"{cfg["act_size"]}",
        cfg["sae_type"].lower(),
        f"{cfg["top_k"]}",
        f"{cfg["lr"]}",
        f"{cfg["name"]}",
    ]
    
    # Add iteration target if specified
    if hasattr(cfg, 'n_iters') and cfg["n_iters"] is not None:
        components.append(f"{cfg["n_iters"]}")
    
    # Create base directory name
    dir_name = "_".join(components)
    
    # For resumed training, add a resume indicator
    if resume:
        # Extract the original checkpoint name if resuming
        if hasattr(cfg, 'resume_from') and cfg["resume_from"]:
            original_dir = os.path.basename(os.path.dirname(cfg["resume_from"]))
            # Extract iteration from checkpoint filename
            iter_match = re.search(r'checkpoint_(\d+)', os.path.basename(cfg["resume_from"]))
            resume_iter = iter_match.group(1) if iter_match else "unknown"
            
            # Count previous resumes to create a compact name
            resume_count = len(cfg.get('resume_history', []))
            
            # Create resumed directory name
            dir_name = f"{original_dir}_resumed{resume_count}_{resume_iter}_to_{cfg["n_iters"]}"
        else:
            # Fallback if resume_from not specified properly
            dir_name = f"{dir_name}_resumed"
    
    return dir_name





def train_sae(
    model,
    cfg,
    hook_point,
    checkpoint_path=None,
    resume=False,
    wandb_run=None,
    model_diffing=False,
    paths=None,
    **kwargs
):
    """
    Train a sparse autoencoder with support for resuming training.
    
    Args:
        model: The language model
        cfg: Training configuration
        hook_point: Hook point for extracting activations
        checkpoint_path: Optional path to checkpoint for resuming
        resume: Whether to resume training
        wandb_run: Optional wandb run
        paths: Optional paths configuration
    """
    # Resolve paths using the paths configuration system
    if paths is None:
        paths = resolve_paths(cfg)
    
    # Set up devices
    device = cfg["device"]
    
    # Initialize SAE and optimizer
    sae = None
    optimizer = None
    scheduler = None
    activation_store = None
    start_iter = 0

    # Handle resuming from checkpoint
    if resume and checkpoint_path:
        # Modify config to include resume information
        cfg["resume_from"] = checkpoint_path
        
        # Create activation store (will be populated from checkpoint)
        activation_store = ActivationsStore(model, cfg)

        sae = BatchTopKSAE(cfg)
        
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])
        
        # Initialize scheduler if specified
        if cfg.get("use_scheduler", False):
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg.get("scheduler_class", "ReduceLROnPlateau"))
            scheduler = scheduler_class(optimizer, **cfg.get("scheduler_args", {}))
        
        # Load checkpoint
        sae, optimizer, cfg_checkpoint, start_iter, scheduler, _ = load_checkpoint(
            checkpoint_path,
            sae=sae,
            optimizer=optimizer,
            scheduler=scheduler,
            activation_store=activation_store,
            device=device
        )

        if model_diffing:
            start_iter = 0

        decoder_weights = sae.state_dict()["W_dec"].cpu()
        
        print(f"Resuming training from iteration {start_iter}")
    else:
        # Normal initialization for new training
        sae = BatchTopKSAE(cfg)
        
        optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])
        
        # Initialize scheduler if specified
        if cfg.get("use_scheduler", False):
            scheduler_class = getattr(torch.optim.lr_scheduler, cfg.get("scheduler_class", "ReduceLROnPlateau"))
            scheduler = scheduler_class(optimizer, **cfg.get("scheduler_args", {}))
        
        # Initialize activation store
        activation_store = ActivationsStore(model, cfg)

    # Generate checkpoint directory
    checkpoint_dir = os.path.join(
        paths["checkpoints_dir"],
        generate_checkpoint_dir(cfg, resume=resume)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "percentiles"), exist_ok=True)
    
    # Initialize feature activation stats tracking
    n_features = sae.state_dict()["W_dec"].shape[0]
    feature_min_activations_buffer = []
    threshold_compute_freq = cfg.get("threshold_compute_freq", 1000)  # How often to compute thresholds
    threshold_num_batches = cfg.get("threshold_num_batches", 20)  # How many batches to collect before computing
    
    for iter_num in range(start_iter, cfg["n_iters"]):
        print(iter_num)        # Process batch
        batch = activation_store.next_batch()
        sae_output = sae(batch)
        loss = sae_output["loss"]

        # Collect feature activations for threshold computation
        if iter_num % threshold_compute_freq == 0 and len(feature_min_activations_buffer) < threshold_num_batches:
            feature_activations = sae_output["feature_acts"]
            # For each feature, get the minimum activation that is greater than zero
            filtered_activations = torch.where(feature_activations > 0, feature_activations, float('inf'))
            feature_min_activations = torch.min(filtered_activations, dim=0).values
            feature_min_activations = torch.where(feature_min_activations == float('inf'), torch.nan, feature_min_activations)
            
            # Store these minimum activations for later processing
            feature_min_activations_buffer.append(feature_min_activations.detach())
            
            # If we've collected enough batches, compute the thresholds
            if len(feature_min_activations_buffer) >= threshold_num_batches:
                with torch.no_grad():
                    # Stack all collected minimum activations
                    all_feature_min_activations = torch.stack(feature_min_activations_buffer)
                    
                    # Replace any remaining infs with nans for proper quantile calculation
                    all_feature_min_activations = torch.where(
                        all_feature_min_activations == float('inf'), 
                        torch.nan, 
                        all_feature_min_activations
                    )
                    
                    # Compute percentiles (0 to 100 in steps of 1)
                    percentiles = torch.linspace(0, 1, 101, device=device)
                    feature_percentiles = torch.nanquantile(all_feature_min_activations, percentiles, dim=0)
                    
                    # Save each percentile
                    for i, percentile in enumerate(feature_percentiles):
                        torch.save(percentile.cpu(), f"{checkpoint_dir}/percentiles/feature_percentile_{i}.pt")
                    
                    # Compute thresholds as mean of non-nan percentiles
                    feature_thresholds = torch.nanmean(feature_percentiles, dim=0)
                    
                    # Save the thresholds
                    torch.save(feature_thresholds.cpu(), f"{checkpoint_dir}/thresholds.pt")
                    
                    # Log statistics if wandb is enabled
                    if wandb_run is not None:
                        # Calculate statistics on thresholds
                        valid_thresholds = feature_thresholds[~torch.isnan(feature_thresholds)]
                        if len(valid_thresholds) > 0:
                            wandb_run.log({
                                "thresholds/mean": torch.mean(valid_thresholds).item(),
                                "thresholds/median": torch.median(valid_thresholds).item(),
                                "thresholds/min": torch.min(valid_thresholds).item(),
                                "thresholds/max": torch.max(valid_thresholds).item(),
                                "thresholds/std": torch.std(valid_thresholds).item(),
                                "thresholds/active_features": len(valid_thresholds),
                            }, step=iter_num)
                    
                    # Clear the buffer for next round
                    feature_min_activations_buffer = []
                    
                    # Free up GPU memory
                    del all_feature_min_activations, feature_percentiles
                    torch.cuda.empty_cache()

        # Logging and checkpointing
        if wandb_run is not None:
            if iter_num % cfg["perf_log_freq"] == 0:
                log_wandb(sae_output, iter_num, wandb_run)
                log_decoder_weights(sae, decoder_weights, iter_num, wandb_run)
        if iter_num % cfg["checkpoint_freq"] == 0 and iter_num > 0:
            save_checkpoint(
                sae, optimizer, cfg, iter_num, checkpoint_dir, 
                scheduler=scheduler, activation_store=activation_store
            )
            
        # Update activation store position
        activation_store.update_position(
            (activation_store.current_batch_idx + 1) % cfg["model_batch_size"],
            activation_store.current_epoch + ((activation_store.current_batch_idx + 1) // cfg["model_batch_size"])
        )

        loss.backward()
        
        # Log gradient norms and learning rate before optimizer step
        if iter_num % cfg["perf_log_freq"] == 0 and wandb_run is not None:
            # Calculate and log gradient norm
            total_grad_norm = 0.0
            for p in sae.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to wandb
            if wandb_run is not None:
                wandb_run.log({
                    "training/gradient_norm": total_grad_norm,
                    "training/learning_rate": current_lr
                }, step=iter_num)
        
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()
        # Update scheduler if used
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
    
    # Save final checkpoint
    save_checkpoint(
        sae, optimizer, cfg, cfg["n_iters"], checkpoint_dir, 
        scheduler=scheduler, activation_store=activation_store, is_final=True
    )    
     
    return sae, checkpoint_dir

def train_sae_group(saes, activation_store, model, cfgs):
    num_batches = cfgs[0]["num_tokens"] // cfgs[0]["batch_size"]
    optimizers = [torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])) for sae, cfg in zip(saes, cfgs)]
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfgs[0])

    batch_tokens = activation_store.get_batch_tokens()

    for i in pbar:
        batch = activation_store.next_batch()
        counter = 0
        for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_wandb(sae_output, i, wandb_run, index=counter)
            if i % cfg["perf_log_freq"]  == 0:
                log_model_performance(wandb_run, i, model, activation_store, sae, index=counter, batch_tokens=batch_tokens)

            if i % cfg["checkpoint_freq"] == 0:
                save_checkpoint(cfg["checkpoint_dir"],wandb_run, sae, cfg, i)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "L0": f"{sae_output['l0_norm']:.4f}", "L2": f"{sae_output['l2_loss']:.4f}", "L1": f"{sae_output['l1_loss']:.4f}", "L1_norm": f"{sae_output['l1_norm']:.4f}"})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
            sae.make_decoder_weights_and_grad_unit_norm()
            optimizer.step()
            optimizer.zero_grad()
            counter += 1
   
    for sae, cfg, optimizer in zip(saes, cfgs, optimizers):
        save_checkpoint(cfg["checkpoint_dir"],wandb_run, sae, cfg, i)

def get_sae_class(sae_type):
    """
    Get the SAE class based on the SAE type string.
    
    Args:
        sae_type: String identifier for the SAE type
        
    Returns:
        SAE class
        
    Raises:
        ValueError: If the SAE type is not recognized
    """
    sae_classes = {
        "BatchTopKSAE": BatchTopKSAE,
        "TopKSAE": TopKSAE,
        "VanillaSAE": VanillaSAE,
        "JumpReLUSAE": JumpReLUSAE
    }
    
    # Allow case-insensitive matching
    sae_type_lower = sae_type.lower()
    for class_name, class_obj in sae_classes.items():
        if class_name.lower() == sae_type_lower:
            return class_obj
    
    # If we get here, the SAE type wasn't found
    raise ValueError(f"Unrecognized SAE type: {sae_type}. Available types: {list(sae_classes.keys())}")

