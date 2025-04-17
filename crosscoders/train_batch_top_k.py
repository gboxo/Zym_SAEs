# %%
from utils import *
from trainer_batch_topk import Trainer
from utils_gb import get_ht_model, load_model

# %%
device = 'cuda:0'


#base_path = "AI4PD/ZymCTRL"

base_path = "/users/nferruz/gboxo/models/ZymCTRL/"
dpo_path = "/users/nferruz/gboxo/output_iteration29/"


tokenizer,base_model = load_model(base_path)
dpo_tokenizer,dpo_model = load_model(dpo_path)

base_model_config = base_model.config
base_model_config.attn_implementation = "eager"
base_model_config.d_model = 5120
base_model = get_ht_model(base_model, base_model_config)

dpo_model_config = dpo_model.config
dpo_model_config.attn_implementation = "eager"
dpo_model_config.d_model = 5120
dpo_model = get_ht_model(dpo_model, dpo_model_config)


all_tokens = load_brenda_tokens()

# %%
default_cfg = {
    "top_k_aux":512,
    "n_batches_to_dead": 5,
    "seed": 49,
    "top_k":32,
    "batch_size": 512,
    "buffer_mult": 4*128,
    "lr": 5e-5,
    "num_tokens": 400_000_000,
    "l1_coeff": 2,
    "aux_penalty": 0.032,
    "beta1": 0.9,
    "beta2": 0.999,
    "d_in": 1280,
    "dict_size": 12*1280,
    "seq_len": 512,
    "enc_dtype": "fp32",
    "model_name": "ZymCTRL",
    "site": "resid_pre",
    "device": "cuda:0",
    "model_batch_size": 128,
    "log_every": 100,
    "save_every": 5000,
    "dec_init_norm": 0.08,
    "hook_point": "blocks.15.hook_resid_pre",
    "wandb_project": "crosscoder-model-diff",
    "wandb_entity": "mi_gbc",
}
cfg = arg_parse_update_cfg(default_cfg)

trainer = Trainer(cfg, base_model, dpo_model, all_tokens)
trainer.train()
# %%
