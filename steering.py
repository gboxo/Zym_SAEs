import torch
import torch.nn as nn
import torch.optim as optim
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import pandas as pd
import os
import pickle as pkl
from tqdm import tqdm

model_iteration = 9
data_iteration = 9
cs = torch.load("/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/all_cs.pt")
cs = cs[f"M{model_iteration}_D{data_iteration}_vs_M0_D0"].cpu().numpy()
# Load the dataframe
model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase New/M{model_iteration}_D{data_iteration}/diffing/"
cfg_sae, sae = load_sae(sae_path)
thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
state_dict = sae.state_dict()
state_dict["threshold"] = thresholds
del sae

cfg = SAEConfig(
    architecture="jumprelu",
    d_in=1280,
    d_sae=10240,
    activation_fn_str="relu",
    apply_b_dec_to_input=True,
    finetuning_scaling_factor=False,
    context_size=256,
    model_name="ZymCTRL",
    hook_name="blocks.26.hook_resid_pre",
    hook_layer=26,
    hook_head_index=None,
    prepend_bos=False,
    dtype="float32",
    normalize_activations="layer_norm",
    dataset_path=None,
    dataset_trust_remote_code=False,
    device="cuda",
    sae_lens_training_version=None,
)


sae = SAE(cfg)
sae.load_state_dict(state_dict)

tokenizer, model = load_model(model_path)
model = get_sl_model(model, model.config, tokenizer).to("cuda")

path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/important_features/important_features_M{model_iteration}_D{data_iteration}.pkl"
with open(path, "rb") as f:
    important_features = pkl.load(f)
feature_indices = important_features["unique_coefs"]

print(important_features["coefs"])
print(feature_indices)