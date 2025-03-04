import argparse
from datetime import datetime
from ..utils import load_config, load_model
from ..config.paths import add_path_args, resolve_paths
from .training import train_sae
from .logs import init_wandb, load_checkpoint
from .sae import BatchTopKSAE
from .activation_store import ActivationsStore
from .config import get_default_cfg, post_init_cfg
from ..utils import get_ht_model
import torch

def main():
    parser = argparse.ArgumentParser()
    parser = add_path_args(parser)
    
    # Add training arguments
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--additional_iters", type=int, help="Additional iterations to train when resuming")
    
    args = parser.parse_args()
    
    # Resolve paths
    paths = resolve_paths(args)
    
    # Load config
    cfg = load_config(args.config)
    cfg["use_wandb"] = True
    
    # Update paths in config
    
    # Handle resuming
    if args.resume:
        if not args.checkpoint_path:
            raise ValueError("Must provide --checkpoint_path when using --resume")
            
        # If checkpoint_path is a directory, it will be handled by load_checkpoint
        checkpoint_path = args.checkpoint_path
        
        # Load checkpoint to get basic info without modifying anything
        _, _, loaded_cfg, start_iter, _, _ = load_checkpoint(checkpoint_path, device=cfg["device"])
        
        # Update iterations if additional_iters is specified
        if args.additional_iters:
            cfg["n_iters"] = start_iter + args.additional_iters
        else:
            # Use original target iterations if not specified
            cfg["n_iters"] = loaded_cfg["n_iters"]
        
        if "model_type" not in cfg:
            cfg["model_type"] = loaded_cfg["model_type"]
        
        # Initialize wandb with resumed config
        wandb_run = init_wandb(cfg, resume=True)
        print("Resuming training from checkpoint")
        print(wandb_run)
        
        # Load model
        tokenizer, model = load_model(cfg["model_path"])
        config = model.config
        config.attn_implementation = "eager"
        config.d_model = 5120
        model = get_ht_model(model, config)


        
        
        # Train with resume
        sae, checkpoint_dir = train_sae(
            model=model,
            cfg=cfg,
            hook_point=cfg["hook_point"],
            checkpoint_path=checkpoint_path,
            resume=True,
            wandb_run=wandb_run,
        )
    else:
        # Normal training
        wandb_run = init_wandb(cfg)
        tokenizer, model = load_model(cfg["model_path"])
        config = model.config
        config.attn_implementation = "eager"
        config.d_model = 5120
        model = get_ht_model(model, config)
        
        sae, checkpoint_dir = train_sae(
            model=model,
            cfg=cfg,
            hook_point=cfg["hook_point"],
            wandb_run=wandb_run,
        )
    
    print(f"Training completed. Checkpoint saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()







