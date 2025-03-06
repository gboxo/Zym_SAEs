from src.utils import load_config, load_model
from src.config.paths import add_path_args, resolve_paths
from src.training.logs import init_wandb, load_checkpoint
from src.training.sae import BatchTopKSAE
from src.training.activation_store import ActivationsStore
from src.utils import get_ht_model
import torch




config = "configs/workstation.yaml"

checkpoint_path ="/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000"
additional_iters = 10
cfg = load_config(config)
cfg["use_wandb"] = True

_, _, loaded_cfg, start_iter, _, _ = load_checkpoint(checkpoint_path, device=cfg["device"])
cfg["n_iters"] = start_iter + additional_iters
#wandb_run = init_wandb(cfg, resume=True)



# =======


tokenizer, model = load_model(cfg["model_path"])
config = model.config
config.attn_implementation = "eager"
config.d_model = 5120
model = get_ht_model(model, config)
paths = resolve_paths(cfg)
cfg["dtype"] = torch.float32

cfg["batch_size"] = 32
cfg["model_batch_size"] = 64 

import numpy as np
activation_store = ActivationsStore(model, cfg)
dataset = activation_store.dataset
sequences = [next(dataset)["input_ids"] for _ in range(128000)]
tot = [sum(elem) for elem in sequences]
tot = np.array(tot)
sequences = np.array(sequences)
ids = np.where(tot < 1000)[0]









batch = activation_store.next_batch()


batch.requires_grad = True
batch.retain_grad()
sae = BatchTopKSAE(cfg)
# Initialize optimizer
optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"])
sae_output = sae(batch)
loss = sae_output["loss"]
print("Loss",loss.requires_grad)
