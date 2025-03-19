import torch
from .logs import log_wandb, log_model_performance, save_checkpoint, load_checkpoint, log_decoder_weights
import os
import re
from datetime import datetime
from .activation_store import ActivationsStore
from ..training.sae import BatchTopKSAE
from types import SimpleNamespace



def generate_checkpoint_dir(cfg: SimpleNamespace, resume: bool = False,diffing: bool = False):
    """
    Generate a checkpoint directory name that clearly indicates original properties
    and continuation without being too long.
    """
    # Extract base model name
    model_name = os.path.basename(cfg.base.model_path.rstrip('/'))
    
    # Format date
    date_str = datetime.now().strftime("%d_%m_%y")
    
    # Base components of the name
    components = [
        model_name,
        date_str,
        f"{cfg.sae.layer}",
        f"{cfg.sae.site}",
        f"{cfg.sae.act_size}",
        cfg.sae.model_type.lower(),
        f"{cfg.training.top_k}",
        f"{cfg.training.lr}",
        f"{cfg.training.name}",
    ]
    
    
    # Create base directory name
    dir_name = "_".join(components)
    
    # For resumed training, add a resume indicator
    if resume:
        # Extract the original checkpoint name if resuming
        if cfg.resuming.resume_from:

            original_dir = os.path.basename(cfg.resuming.resume_from)
            # Extract iteration from checkpoint filename
            iter_match = re.search(r'checkpoint_(\d+)', os.path.basename(cfg.resuming.resume_from))
            resume_iter = iter_match.group(1) if iter_match else "X"
            
            
            # Create resumed directory name
            dir_name = f"{original_dir}_resumed{resume_iter}_to_{cfg.resuming.n_iters}"
        else:
            # Fallback if resume_from not specified properly
            dir_name = f"{dir_name}_resumed"
    if diffing:
        dir_name = f"diffing"
    else:
        dir_name = f"final"
    
    return dir_name


def threshold_loop_collect(sae_output, feature_min_activations_buffer):
    with torch.no_grad():
        feature_activations = sae_output["feature_acts"]
        # For each feature, get the minimum activation that is greater than zero
        filtered_activations = torch.where(feature_activations > 0, feature_activations, float('inf'))
        feature_min_activations = torch.min(filtered_activations, dim=0).values
        feature_min_activations = torch.where(feature_min_activations == float('inf'), torch.nan, feature_min_activations)
        feature_min_activations_buffer.append(feature_min_activations.detach())
    return feature_min_activations_buffer


def threshold_loop_compute(feature_min_activations_buffer, checkpoint_dir):
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
        percentiles = torch.linspace(0, 1, 101)
        feature_percentiles = torch.nanquantile(all_feature_min_activations, percentiles, dim=0)
        
        # Save each percentile
        for i, percentile in enumerate(feature_percentiles):
            torch.save(percentile.cpu(), f"{checkpoint_dir}/percentiles/feature_percentile_{i}.pt")
        
        # Compute thresholds as mean of non-nan percentiles
        feature_thresholds = torch.nanmean(feature_percentiles, dim=0)
        
        # Save the thresholds
        torch.save(feature_thresholds.cpu(), f"{checkpoint_dir}/thresholds.pt")
    return feature_thresholds




def get_gradient_norm(sae, optimizer):
    with torch.no_grad():
        # Calculate and log gradient norm
        total_grad_norm = 0.0
        for p in sae.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_grad_norm += param_norm ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
    return total_grad_norm, current_lr

def validate_and_clip_gradients(sae, optimizer, max_grad_norm, iter_num=None, wandb_run=None):
    """Helper function to validate gradients and perform clipping."""
    # Check if any gradients are invalid (inf/nan)
    if not all(torch.isfinite(p.grad).all() for p in sae.parameters() if p.grad is not None):
        print(f"Warning: Non-finite gradients detected at iteration {iter_num}")
        if wandb_run is not None:
            wandb_run.log({"training/invalid_gradients": 1}, step=iter_num)
        optimizer.zero_grad()
        return False
    
    # Clip gradients and normalize decoder weights
    torch.nn.utils.clip_grad_norm_(sae.parameters(), max_grad_norm)
    sae.make_decoder_weights_and_grad_unit_norm()
    return True

def resume_training(
    model,
    cfg: SimpleNamespace,
    sae_cfg: dict,
    checkpoint_path: str = None,
    resume: bool = False,
    wandb_run = None,
    **kwargs
):
    # Set up devices
    device = cfg.sae.device


    sae = BatchTopKSAE(sae_cfg)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.training.lr, amsgrad=True)

    
    
    # Load checkpoint
    sae, optimizer, cfg_checkpoint, start_iter, activation_store_state = load_checkpoint(
        sae = sae,
        checkpoint_path=checkpoint_path,
        optimizer=optimizer,
        device=device
    )


    if cfg.resuming.model_diffing:
        start_iter = 0
        activation_store_state = None

    decoder_weights = sae.state_dict()["W_dec"].cpu()
    

    

    # Initialize activation store
    activation_store = ActivationsStore(model, sae_cfg, activation_store_state)

    # Generate checkpoint directory
    checkpoint_dir = os.path.join(
        cfg.resuming.checkpoint_dir_to,
        generate_checkpoint_dir(cfg, resume=resume, diffing=cfg.resuming.model_diffing)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "percentiles"), exist_ok=True)
    
    # Initialize feature activation stats tracking
    feature_min_activations_buffer = []
    threshold_compute_freq = cfg.training.threshold_compute_freq  # How often to compute thresholds
    threshold_num_batches = cfg.training.threshold_num_batches  # How many batches to collect before computing


    # Compute the number of iterations
    start_iter = start_iter // cfg.training.batch_size
    n_tokens = cfg.training.num_tokens
    n_iters = n_tokens // cfg.training.batch_size
    print("Number of iterations: ", n_iters)
    print("Starting training loop")
    
    accumulation_steps = 4    
    for iter_num in range(start_iter, start_iter + n_iters):
        print("===========")
        sae.train()
        print("Iteration: ", iter_num)
        
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of accumulation
        
        # Accumulate gradients over multiple batches
        for acc_step in range(accumulation_steps):
            batch = activation_store.next_batch()
            sae_output = sae(batch)
            loss = sae_output["loss"] / accumulation_steps  # Scale loss by accumulation steps
            total_loss += loss.item()
            
            loss.backward()
        
        # Update weights after accumulation
        if validate_and_clip_gradients(sae, optimizer, cfg.training.max_grad_norm, iter_num, wandb_run):
            optimizer.step()
        optimizer.zero_grad()
        
        print(f"Average Loss over {accumulation_steps} batches: ", total_loss / accumulation_steps)

        # Validation and logging
        if iter_num % cfg.training.perf_log_freq == 0 and wandb_run is not None:
            print("Logging WandB and decoder  weights (if diffing)")
            # Switch to eval mode for validation
            sae.eval()
            with torch.no_grad():
                # Log training metrics
                log_wandb(sae_output, iter_num, wandb_run)
                
                if cfg.resuming.model_diffing:
                    log_decoder_weights(sae, decoder_weights, iter_num, wandb_run)
                
                
                # Get gradient stats
                total_grad_norm, current_lr = get_gradient_norm(sae, optimizer)
                
                # Log training stats
                wandb_run.log({
                    "training/gradient_norm": total_grad_norm,
                    "training/learning_rate": current_lr
                }, step=iter_num)
            
            # Switch back to training mode
            sae.train()

        # Threshold computation
        if iter_num % threshold_compute_freq == 0 and len(feature_min_activations_buffer) < threshold_num_batches:
            print("Collecting thresholds")
            sae.eval()
            with torch.no_grad():
                feature_min_activations_buffer = threshold_loop_collect(sae_output, feature_min_activations_buffer)
                if len(feature_min_activations_buffer) >= threshold_num_batches:
                    print("Computing thresholds")
                    feature_thresholds = threshold_loop_compute(feature_min_activations_buffer, checkpoint_dir)
                    if wandb_run is not None:
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
                print("Resetting feature min activations buffer")
                feature_min_activations_buffer = []
            sae.train()


            
        # Update activation store position
        activation_store.update_position(
            activation_store.current_batch_idx + 1,
            activation_store.current_epoch
        )

        if iter_num % cfg.training.checkpoint_freq == 0 and iter_num > 0:
            print("Saving checkpoint")
            save_checkpoint(
                sae, optimizer, cfg, iter_num, checkpoint_dir, 
                 activation_store=activation_store
            )


    # Save final checkpoint
    save_checkpoint(
        sae, optimizer, cfg, start_iter + n_iters, checkpoint_dir, 
        activation_store=activation_store, is_final=True
    )    
     
    return sae, checkpoint_dir




def train_sae(
    model,
    cfg: SimpleNamespace,
    sae_cfg: dict,
    checkpoint_path: str = None,
    wandb_run = None,
    **kwargs
):
    """
    Train a sparse autoencoder with support for resuming training.
    
    Args:
        model: The language model
        cfg: Training configuration
        sae_cfg: SAE configuration
        checkpoint_path: Optional path to checkpoint for resuming
        resume: Whether to resume training
        wandb_run: Optional wandb run
    """
    print("Starting training")
    
    # Set up devices
    device = cfg.sae.device
    
    # Initialize SAE and optimizer
    sae = None
    optimizer = None
    activation_store = None
    start_iter = 0

    # Handle resuming from checkpoint
    # Normal initialization for new training
    sae = BatchTopKSAE(sae_cfg)
    print("SAE initialized")
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.training.lr, amsgrad=True)
    print("Optimizer initialized")
    
    
    # Initialize activation store
    activation_store = ActivationsStore(model, sae_cfg)
    print("Activation store initialized")
    # Generate checkpoint directory
    checkpoint_dir = os.path.join(
        cfg.training.checkpoint_dir,
        generate_checkpoint_dir(cfg, resume=False)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir, "percentiles"), exist_ok=True)
    print("Checkpoint directory created")
    # Initialize feature activation stats tracking
    n_features = sae.state_dict()["W_dec"].shape[0]
    feature_min_activations_buffer = []
    threshold_compute_freq = cfg.training.threshold_compute_freq  # How often to compute thresholds
    threshold_num_batches = cfg.training.threshold_num_batches  # How many batches to collect before computing



    n_tokens = cfg.training.num_tokens
    n_iters = n_tokens // cfg.training.batch_size
    print("Number of iterations: ", n_iters)
    print("Starting training loop")

    accumulation_steps = 4  # Number of batches to accumulate
    print(f"Accumulating gradients over {accumulation_steps} steps")
    
    for iter_num in range(n_iters):
        print("===========")
        sae.train()
        print("Iteration: ", iter_num)
        
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of accumulation
        
        # Accumulate gradients over multiple batches
        for acc_step in range(accumulation_steps):
            batch = activation_store.next_batch()
            sae_output = sae(batch)
            loss = sae_output["loss"] / accumulation_steps  # Scale loss by accumulation steps
            total_loss += loss.item() * accumulation_steps  # Multiply by accumulation_steps to get true loss
            
            loss.backward()
        
        # Update weights after accumulation
        if validate_and_clip_gradients(sae, optimizer, cfg.training.max_grad_norm, iter_num, wandb_run):
            optimizer.step()
        optimizer.zero_grad()
        
        print(f"Average Loss over {accumulation_steps} batches: ", total_loss / accumulation_steps)

        # Validation and logging
        if iter_num % cfg.training.perf_log_freq == 0 and wandb_run is not None:
            print("Logging WandB")
            # Switch to eval mode for validation
            sae.eval()
            with torch.no_grad():
                # Log training metrics
                log_wandb(sae_output, iter_num, wandb_run)
                
                
                # Log validation metrics
                # Get gradient stats
                total_grad_norm, current_lr = get_gradient_norm(sae, optimizer)
                
                # Log training stats
                wandb_run.log({
                    "training/gradient_norm": total_grad_norm,
                    "training/learning_rate": current_lr
                }, step=iter_num)
            
            # Switch back to training mode
            sae.train()

        # Threshold computation
        if iter_num % threshold_compute_freq == 0 and len(feature_min_activations_buffer) < threshold_num_batches:
            print("Collecting thresholds")
            sae.eval()
            with torch.no_grad():
                feature_min_activations_buffer = threshold_loop_collect(sae_output, feature_min_activations_buffer)
                if len(feature_min_activations_buffer) >= threshold_num_batches:
                    print("Computing thresholds")
                    feature_thresholds = threshold_loop_compute(feature_min_activations_buffer, checkpoint_dir)
                    if wandb_run is not None:
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
                print("Resetting feature min activations buffer")
                feature_min_activations_buffer = []
            sae.train()

            
        # Update activation store position
        activation_store.update_position(
            activation_store.current_batch_idx + 1,
            activation_store.current_epoch
        )

        if iter_num % cfg.training.checkpoint_freq == 0 and iter_num > 0:
            print("Saving checkpoint")
            save_checkpoint(
                sae, optimizer, cfg, iter_num, checkpoint_dir, 
                 activation_store=activation_store
            )
    # Save final checkpoint
    save_checkpoint(
        sae, optimizer, cfg, iter_num, checkpoint_dir, 
        activation_store=activation_store, is_final=True
    )    
     
    return sae, checkpoint_dir