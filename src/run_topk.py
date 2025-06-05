import argparse
from utils import load_model
from training.training import train_sae, resume_training
from training.logs import init_wandb
from utils import get_ht_model
from config.load_config import load_experiment_config, convert_to_sae_config
from types import SimpleNamespace

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/simple_training.yaml")
    args = parser.parse_args()


    config = load_experiment_config(args.config)
    sae_cfg = convert_to_sae_config(config)
    # Convert nested dictionaries to nested SimpleNamespace objects
    config = {k: SimpleNamespace(**v) if isinstance(v, dict) else v for k, v in config.items()}
    config = SimpleNamespace(**config)   

    wandb_run = init_wandb(config)
    tokenizer, model = load_model(config.base.model_path)
    print("The model is loaded")
    model_config = model.config
    model_config.attn_implementation = "eager"
    model_config.d_model = 5120
    model = get_ht_model(model, model_config)
    print("The model is loaded as TL")
    if not config.resuming.resuming:
    
        sae, checkpoint_dir = train_sae(
            model=model,
            cfg=config,
            sae_cfg=sae_cfg,
            wandb_run=wandb_run,
        )
    else:
        sae, checkpoint_dir = resume_training(
            model=model,
            cfg=config,
            sae_cfg=sae_cfg,
            wandb_run=wandb_run,
            resume=True,
            checkpoint_path=config.resuming.resume_from,
        )

    print(f"Training completed. Checkpoint saved to {checkpoint_dir}")

if __name__ == "__main__":
    main()







