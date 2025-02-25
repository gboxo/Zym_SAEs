import torch
from torch.nn import functional as F
from BatchTopK.sae import BatchTopKSAE
from BatchTopK.config import get_default_cfg, post_init_cfg
from weight_conversion import get_ht_model
from BatchTopK.activation_store import ActivationsStore
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
model_ht = AutoModelForCausalLM.from_pretrained("nferruz/ProtGPT2",
                                                attn_implementation="eager",
                                                torch_dtype=torch.float32)
config = model_ht.config
config.d_mlp = 5120



model = get_ht_model(model_ht,config, tokenizer=tokenizer)
del model_ht



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
activations_store = ActivationsStore(model, cfg)


input_ids = activations_store.get_batch_tokens()
input_ids = input_ids[:4,]






train_path = "/users/nferruz/gboxo/BatchTopK/checkpoints/ProtGPT2_blocks.10.hook_resid_pre_5120_batchtopk_32_0.0003_1953124/"
sae_weights = torch.load(train_path+"sae.pt", weights_only=False, map_location='cpu')
sae = BatchTopKSAE(cfg)
sae.load_state_dict(sae_weights)



logits, cache = model.run_with_cache(input_ids, names_filter= lambda x: "10.hook_resid_pre" in x)



acts = cache["blocks.10.hook_resid_pre"]
acts = acts[0]
reconstruction = sae(acts)
x, mean, std = sae.preprocess_input(acts)

x_cent = x - sae.b_dec
acts = F.relu(x_cent @ sae.W_enc)
acts_topk = torch.topk(acts.flatten(), sae.cfg["top_k"] * x.shape[0], dim=-1)
acts_topk = (
    torch.zeros_like(acts.flatten())
    .scatter(-1, acts_topk.indices, acts_topk.values)
    .reshape(acts.shape)
)
    
x_reconstruct = acts_topk @ sae.W_dec + sae.b_dec

sae_out = reconstruction["sae_out"]
sae_out_prenorm = (sae_out - mean)/std


l2_loss = (sae_out_prenorm.float() - x.float()).pow(2).mean().item()




