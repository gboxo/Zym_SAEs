import argparse
from src.training.training import train_sae
from src.training.logs import init_wandb, load_checkpoint
from src.utils import get_ht_model
from src.utils import load_config, load_model
from src.config.paths import add_path_args

import torch

def main():
    
    # Add training arguments
    config = "configs/workstation.yaml"
    checkpoint_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_04_03_25_hhook_resid_pre_1280_batchtopk_100_0.0003_resumed/checkpoint_latest.pt"
    dataset_path = "Data/Diffing/tokenized_train_dataset_iteration1"

    
    
    
    # Load config
    cfg = load_config(config)
    cfg["use_wandb"] = True
    
    #wandb_run = init_wandb(cfg, resume=True)
    # If checkpoint_path is a directory, it will be handled by load_checkpoint
    
    # Load checkpoint to get basic info without modifying anything
    _, _, loaded_cfg, start_iter, _, _ = load_checkpoint(checkpoint_path, device=cfg["device"])
    
    # Update iterations if additional_iters is specified
    
    if "model_type" not in cfg:
        cfg["model_type"] = loaded_cfg["model_type"]
    
    # Initialize wandb with resumed config
    print("Resuming training from checkpoint")

    # Load model
    tokenizer, model = load_model(cfg["model_path"])
    config = model.config
    config.attn_implementation = "eager"
    config.d_model = 5120
    model = get_ht_model(model, config)


    loaded_cfg["name"] = "Model_Diffing_M0_D1"
    loaded_cfg["dtype"] = torch.float32
    loaded_cfg["wandb_project"] = "Model_Diffing" 
    loaded_cfg["batch_size"] = 512
    loaded_cfg["top_k"] = 100
    loaded_cfg["aux_penalty"] = 0.01
    loaded_cfg["top_k_aux"] = 512
    loaded_cfg["model_batch_size"] = 128 
    loaded_cfg["dataset_path"] = dataset_path
    loaded_cfg["perf_log_freq"] = 1
    loaded_cfg["model_name"] = cfg["model_name"]
    loaded_cfg["num_batches_in_buffer"] = 5 
    loaded_cfg["n_iters"] = 80
    loaded_cfg["model_type"] = "BatchTopKSAE"
    loaded_cfg["use_wandb"] = True
    loaded_cfg["num_tokens"] = 256*152
    loaded_cfg["checkpoint_dir"] = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/Diffing/checkpoints/"
    loaded_cfg["threshold_compute_freq"] = 1
    loaded_cfg["threshold_num_batches"] = 40 
    loaded_cfg["lr"] = 5e-4


    if False:



        post_sae, checkpoint_dir = train_sae(
            model=model,
            cfg=loaded_cfg,
            hook_point=cfg["hook_point"],
            checkpoint_path=checkpoint_path,
            model_diffing = True,
            resume=True,
            wandb_run=wandb_run,
        )


    # === EVALUATE ===
    from src.training.activation_store import ActivationsStore
    from src.utils import load_sae
    torch.cuda.empty_cache()

    # === POST TRAINING ===
    checkpoint_dir = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_06_03_25_hhook_resid_pre_1280_batchtopk_100_0.0005_resumed/"

    cfg,post_sae = load_sae(checkpoint_dir)
    post_sae.eval()


    cfg = loaded_cfg
    cfg["use_wandb"] = False
    cfg["model_batch_size"] = 128
    cfg["batch_size"] = 64
    cfg["n_iters"] = 10
    cfg["dataset_path"] = "Data/Diffing/tokenized_eval_dataset_iteration1"

    activation_store = ActivationsStore(model,cfg)
    batch = activation_store.next_batch()

    loss = post_sae(batch)["loss"]
    print("Post-training loss: ", loss)


    # === PRE TRAINING ===
    checkpoint_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
    cfg,sae_pre = load_sae(checkpoint_path)
    sae_pre.eval()
    cfg = loaded_cfg
    cfg["use_wandb"] = False
    cfg["model_batch_size"] = 128
    cfg["batch_size"] = 64
    cfg["n_iters"] = 10
    cfg["dataset_path"] = "Data/Diffing/tokenized_eval_dataset_iteration1"


    activation_store = ActivationsStore(model,cfg)
    batch = activation_store.next_batch()
    loss = sae_pre(batch)["loss"]
    print("Pre-training loss: ", loss)


    









if __name__ == "__main__":
    main()







