import argparse
from src.utils import load_config, load_model
from src.config.paths import add_path_args
from src.training.training import train_sae
from src.training.logs import init_wandb, load_checkpoint
from src.utils import get_ht_model
import torch
from src.config.load_config import load_experiment_config, convert_to_sae_config
from types import SimpleNamespace

def main():

    config = load_experiment_config("configs/base_config_workstation.yaml")
    sae_cfg = convert_to_sae_config(config)
    # Convert nested dictionaries to nested SimpleNamespace objects
    config = {k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()}
    config = SimpleNamespace(**config)   

    wandb_run = init_wandb(config)
    tokenizer, model = load_model(config.base.model_path)
    config = model.config
    config.attn_implementation = "eager"
    config.d_model = 5120
    model = get_ht_model(model, config)
    
    sae, checkpoint_dir = train_sae(
        model=model,
        cfg=config,
        sae_cfg=sae_cfg,
        wandb_run=wandb_run,
    )

    print(f"Training completed. Checkpoint saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()







