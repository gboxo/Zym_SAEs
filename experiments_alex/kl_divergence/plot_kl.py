import pickle
import matplotlib.pyplot as plt
import torch
import numpy as np

out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_kl_divergence.pkl"
#out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M0_kl_divergence.pkl"
#out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/dms_kl_divergence.pkl"
with open(out_path, "rb") as f:
    kl_divergences = pickle.load(f)
kl_divergences = list(kl_divergences.values())

kl_divergences = torch.tensor(kl_divergences)

kl_divergences = kl_divergences.detach().cpu().numpy()


# Create a masked array where 0 values will be masked
masked_kl = np.ma.masked_where(kl_divergences < 0.01, kl_divergences)

fig, ax = plt.subplots(figsize=(10, 8))
# Use a custom colormap that sets masked values to white
cmap = plt.cm.viridis
cmap.set_bad('white')
im = ax.imshow(masked_kl, aspect='auto', cmap=cmap, interpolation='nearest')
plt.colorbar(im, ax=ax, label='KL Divergence')
ax.set_title('KL Divergence Heatmap')
ax.set_xlabel('Token Position')
ax.set_ylabel('Sequence Number')
plt.tight_layout()
plt.savefig('kl_divergence_heatmap_M3.png', dpi=300, bbox_inches='tight')
