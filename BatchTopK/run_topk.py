
from weight_conversion import get_ht_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from training import train_sae
from sae import  BatchTopKSAE
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg


cfg = get_default_cfg()
cfg["batch_size"] = 512
cfg["model_batch_size"] = 128
cfg["sae_type"] = "batchtopk"
cfg["n_batches_to_dead"] = 20
cfg["model_name"] = "ProtGPT2"
cfg["layer"] = 10 
cfg["site"] = "resid_pre"
cfg["dataset_path"] = "nferruz/UR50_2021_04"
cfg["aux_penalty"] = (1/32)
cfg["lr"] = 3e-4
cfg["input_unit_norm"] = True
cfg["top_k"] = 32
cfg["dict_size"] = 1280*4
cfg['wandb_project'] = 'protGPT_SAE'
cfg['l1_coeff'] = 0.
cfg['act_size'] = 1280
cfg['device'] = 'cuda'
cfg['bandwidth'] = 0.001
cfg['top_k'] = 32
sae = BatchTopKSAE(cfg)

cfg = post_init_cfg(cfg)


tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model_ht = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2",
                                                attn_implementation="eager",
                                                torch_dtype=torch.float32)

config = model_ht.config
config.d_mlp = 5120
model = get_ht_model(model_ht,config, tokenizer=tokenizer)
activations_store = ActivationsStore(model, cfg)
train_sae(sae, activations_store, model, cfg)






