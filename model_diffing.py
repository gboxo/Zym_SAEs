import argparse
from src.training.training import resume_training
from src.training.logs import init_wandb, load_checkpoint
from src.utils import get_ht_model
from src.utils import load_model
from src.config.load_config import load_experiment_config, convert_to_sae_config
import torch
from types import SimpleNamespace

def main():

    config = load_experiment_config("configs/model_diffing.yaml")
    sae_cfg = convert_to_sae_config(config)
    # Convert nested dictionaries to nested SimpleNamespace objects
    config = {k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()}
    config = SimpleNamespace(**config)

    
    # Add training arguments
    
    wandb_run = init_wandb(config, resume=True)
    
    # Load checkpoint to get basic info without modifying anything
    _, _, loaded_cfg, start_iter, _, _ = load_checkpoint(config.resuming.resume_from, device=config.sae.device)
    

    tokenizer, model = load_model(config.base.model_path)
    model_config = model.config
    model_config.attn_implementation = "eager"
    model_config.d_model = 5120
    model = get_ht_model(model, model_config)
    config.sae.dtype = torch.float32


    post_sae, checkpoint_dir = resume_training(
        model=model,
        cfg=config,
        sae_cfg=sae_cfg,
        checkpoint_path=config.resuming.resume_from,
        model_diffing = True,
        resume=True,
        wandb_run=wandb_run,
    )



if __name__ == "__main__":
    main()







