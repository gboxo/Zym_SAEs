# %%
import os
import sys

src_path = os.path.abspath(os.path.join('../../'))

# Add it to sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

# %%
import torch
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from src.utils import load_sae
import numpy as np
torch.cuda.empty_cache()
# Get the absolute path to the src directory

# %%

base_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_04_03_25_hhook_resid_pre_1280_batchtopk_100_0.0003_resumed/"
diff_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/Diffing Alpha Amylase/"
diff_paths = os.listdir(diff_path)

# %%
iterations = [int(path.split("_")[1][1]) for path in diff_paths]
argsort_iterations = np.argsort(iterations)
sorted_diff_paths = [diff_path+diff_paths[i]+"/diffing/" for i in argsort_iterations]

# %%

# ========= BASE SAE =======

cfg,sae_base = load_sae(base_path)
sae_base.eval()
base_W_dec = sae_base.W_dec.data
del sae_base

# ========= BASE MODEL RL DATA SAE =======

# %%
all_W_decs = []
for path in sorted_diff_paths:
    cfg,sae = load_sae(path)
    sae.eval()
    W_dec = sae.W_dec.data
    all_W_decs.append(W_dec)


# %%

# Stage wise cosine similarity


cs = []
for i in range(len(all_W_decs)):
    a = all_W_decs[i-1]
    b = all_W_decs[i]
    cs.append(F.cosine_similarity(b, a, dim=1).detach().cpu().numpy())

# Leave out entries that never change
equals_one = []
for c in cs:
    w = np.where(c > 0.99, torch.tensor(1), torch.tensor(0))
    equals_one.append(w)

always = np.all(np.array(equals_one), axis=0)


cs_filtered = []
for i,c in enumerate(cs):
    c = c[~always]
    cs_filtered.append(c)

# %%
# violin plot
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
plt.violinplot(cs_filtered)
plt.xlabel("Stage")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity between stages Filtered")

plt.subplot(1,2,2)
plt.violinplot(cs)
plt.xlabel("Stage")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity between stages")
plt.show()







# %%
# Cosine similarity between a stage and the base model
cs_base = []
for i in range(len(all_W_decs)):
    a = base_W_dec
    b = all_W_decs[i]
    cs_base.append(F.cosine_similarity(b, a, dim=1).detach().cpu().numpy())


# %%
# violin plot
plt.violinplot(cs_base)
plt.xlabel("Stage")
plt.ylabel("Cosine Similarity")
plt.title("Cosine Similarity between stages")
plt.show()

# %% 
# New different features over time




# %%
# Change in diffing features over time

