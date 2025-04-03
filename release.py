import torch
import os
import safetensors
from safetensors.torch import save_model
from src.utils import load_sae


if False:
    for layer in range(5, 40,5):

        path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_{layer}/sae_training_iter_0/final/"
        cfg,sae = load_sae(path)

        save_model(sae, f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_{layer}/sae_training_iter_0/final/checkpoint_latest.safetensors")
else:
    path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_5/sae_training_iter_0/final/checkpoint_latest.safetensors"
    state_dict = safetensors.torch.load_file(path)
    print(state_dict.keys())


