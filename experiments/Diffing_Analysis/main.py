# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F

# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join('../../'))

# Add it to sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

# %%


from src.training.activation_store import ActivationsStore
from src.utils import load_sae
torch.cuda.empty_cache()

# === POST TRAINING ===
checkpoint_dir = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_06_03_25_hhook_resid_pre_1280_batchtopk_100_0.0005_resumed/"
cfg,post_sae = load_sae(checkpoint_dir)
post_sae.eval()


# === PRE TRAINING ===
checkpoint_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg,sae_pre = load_sae(checkpoint_path)
sae_pre.eval()

# %%

W_dec_pre = sae_pre.W_dec.data
W_dec_post = post_sae.W_dec.data



# %%
cosine_similarity = F.cosine_similarity(W_dec_pre, W_dec_post, dim=1)
















