
from weight_conversion import get_ht_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel

import torch
from training import train_sae
from sae import  BatchTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg

model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
cfg = get_default_cfg()
cfg["num_tokens"] = int(2e9)
cfg["model_name"] = "ZymCTRL_25_02_25_h100_RAW"
cfg["sae_type"] = "batchtopk"
cfg["seq_len"] = 256
cfg["layer"] = 26
cfg["batch_size"] = 4096
cfg["model_batch_size"] = 1024
cfg['top_k'] = 100 
cfg["top_k_aux"] = 512
cfg["n_batches_to_dead"] = 5 
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "/home/woody/b114cb/b114cb23/boxo/final_dataset_big/"
cfg["aux_penalty"] = 0
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = False
cfg["dict_size"] = 1280*8
cfg['wandb_project'] = 'ZymCTRL_SAE'
cfg['l1_coeff'] = 0.
cfg['act_size'] = 1280
cfg['device'] = 'cuda'
cfg["checkpoint_freq"] = 10000
cfg["checkpoint_dir"] = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/"
sae = BatchTopKSAE(cfg)

cfg = post_init_cfg(cfg)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model_ht = GPT2LMHeadModel.from_pretrained(model_path,
                                                attn_implementation="eager",
                                                torch_dtype=torch.float32)

config = model_ht.config
config.d_mlp = 5120
model = get_ht_model(model_ht,config, tokenizer=tokenizer)
activations_store = ActivationsStore(model, cfg)
train_sae(sae, activations_store, model, cfg)







