# %%
import torch
import os
import matplotlib.pyplot as plt
import sys
import torch.nn.functional as F
from src.utils import load_sae
import numpy as np
torch.cuda.empty_cache()
# Get the absolute path to the src directory
src_path = os.path.abspath(os.path.join('../../'))

# Add it to sys.path
if src_path not in sys.path:
    sys.path.append(src_path)


# %%

base_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_04_03_25_hhook_resid_pre_1280_batchtopk_100_0.0003_resumed/"
path_sae_M0_D9 = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_07_03_25_hhook_resid_pre_1280_batchtopk_100_0.0005_Model_Diffing_M0_D9_resumed/"
path_sae_M9_D9 = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_07_03_25_hhook_resid_pre_1280_batchtopk_100_0.0005_Model_Diffing_M9_D9_resumed/"


# %%

# ========= BASE SAE =======

cfg,sae_base = load_sae(base_path)
sae_base.eval()
base_W_dec = sae_base.W_dec.data

# ========= BASE MODEL RL DATA SAE =======

cfg,sae_M0_D9 = load_sae(path_sae_M0_D9)
sae_M0_D9.eval()
M0_D9_W_dec = sae_M0_D9.W_dec.data

# ========= RL MODEL RL DATA SAE =======

cfg,sae_M9_D9 = load_sae(path_sae_M9_D9)
sae_M9_D9.eval()
M9_D9_W_dec = sae_M9_D9.W_dec.data



# %%



# %%
M0_D0_cs = F.cosine_similarity(base_W_dec, M0_D9_W_dec, dim=1)
M9_D9_cs = F.cosine_similarity(base_W_dec, M9_D9_W_dec, dim=1)

# %%
import seaborn as sns
import matplotlib.pyplot as plt


points1 =  [  64, 1525, 2140, 3184, 3340, 3832, 3882, 4356, 5062, 5249, 6175,
       6393, 6762, 7247, 7344, 8432, 8507, 8697]




# Set style and figure size
plt.figure(figsize=(10, 8))

# Create scatter plot with smaller points and transparency
plt.scatter(M0_D0_cs.detach().cpu().numpy(), 
           M9_D9_cs.detach().cpu().numpy(),
           alpha=0.4,
           s=20,
           color='#2E86C1')

# Add red circle around point with index 3340
for i in points1:
    x = M0_D0_cs.detach().cpu().numpy()[i]
    y = M9_D9_cs.detach().cpu().numpy()[i]
    circle = plt.Circle((x, y), 0.02, color='red', fill=False, linewidth=2)
    # Add text to the circle
    plt.gca().add_patch(circle)
    plt.text(x, y, str(i), fontsize=10, ha='left', va='bottom', color='black')



# Customize labels and title with better fonts
plt.xlabel("Cosine Similarity with Base Model (NO RL)", 
          fontsize=12, 
          fontweight='bold')
plt.ylabel("Cosine Similarity with Base Model (RL)", 
          fontsize=12,
          fontweight='bold')
plt.title("Feature Similarity Comparison Between Base and RL-Trained Models",
          fontsize=14,
          pad=20)

# Add grid and customize ticks
plt.grid(True, linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)

# Adjust layout and display
plt.tight_layout()
plt.show()











