import argparse
from datetime import datetime
from weight_conversion import get_ht_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel
import os
from utils import load_config, load_model

import torch
from training import train_sae
from sae import BatchTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg

# Add argument parser
parser = argparse.ArgumentParser(description='Run TopK SAE training')
parser.add_argument('--is_alex', action='store_true', help='Is the training running on the alex cluster?')
parser.add_argument('--config', type=str, help='Yaml config file')
parser.add_argument('--model_path', type=str, help='Path to the model')
parser.add_argument('--dataset_path', type=str, help='Path to the dataset')
parser.add_argument('--layer', type=int, help='Layer number to analyze')
parser.add_argument('--dict_size', type=int, help='Dictionary size for SAE')
parser.add_argument('--wandb_dir', type=str, help='Path to the wandb directory')
args = parser.parse_args()

# Load config from file or use defaults
if args.config:
    cfg = load_config(args.config)
else:
    cfg = get_default_cfg()
    # Set environment-specific defaults
    if args.is_alex:
        cfg.update({
            "seq_len": 256,
            "batch_size": 4096,
            "model_batch_size": 1024,
            "dict_size": 1280*8,
            "model_name": "ZymCTRL",
            "checkpoint_dir": "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/",
            "wandb_dir": "/home/woody/b114cb/b114cb23/boxo/"
        })
    else:
        cfg.update({
            "seq_len": 128,
            "batch_size": 4096,
            "model_batch_size": 256,
            "dict_size": 5120,
            "model_name": "ZymCTRL",
            "checkpoint_dir": "/users/nferruz/gboxo/ZymCTRL/checkpoints/",
            "wandb_dir": "/users/nferruz/gboxo/wandb/"
        })

    if args.model_path:
        cfg["model_name"] = "ZymCTRL"
    if args.dataset_path:
        cfg["dataset_path"] = args.dataset_path
    if args.layer is not None:
        cfg["layer"] = args.layer
    if args.dict_size:
        cfg["dict_size"] = args.dict_size
    if args.wandb_dir:
        cfg["wandb_dir"] = args.wandb_dir

    # Set other default configurations
    cfg.update({
        "num_tokens": int(2e9),
        "sae_type": "batchtopk",
        "top_k": 100,
        "top_k_aux": 512,
        "n_batches_to_dead": 5,
        "site": "resid_pre",
        "aux_penalty": 0,
        "lr": 3e-4,
        "input_unit_norm": False,
        "l1_coeff": 0.,
        "act_size": 1280,
        "device": "cuda",
        "checkpoint_freq": 10000
    })

# Add timestamp to checkpoint directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg["checkpoint_dir"] = f"{cfg['checkpoint_dir']}/layer{args.layer}_{timestamp}/"
os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

sae = BatchTopKSAE(cfg)

cfg = post_init_cfg(cfg)


tokenizer, model_ht = load_model(args.model_path)

config = model_ht.config
config.d_mlp = 5120
model = get_ht_model(model_ht,config, tokenizer=tokenizer)
activations_store = ActivationsStore(model, cfg)
train_sae(sae, activations_store, model, cfg)







