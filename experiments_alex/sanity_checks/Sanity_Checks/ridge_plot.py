# %%
import numpy as np
import matplotlib.pyplot as plt 
import torch
from torch.nn import functional as F

# %%
import numpy as np
def compute_cs(W_dec1, W_dec2):
    """
    Compute the cosine similarity between two weight matrices
    """
    cs = F.cosine_similarity(W_dec1, W_dec2, dim=0)
    return cs


base = torch.load("/users/nferruz/gboxo/sae_training_iter_0_100/final/checkpoint_latest.pt")["model_state_dict"]["W_dec"]
cs_list = []
cs_m0_list = []
for i in range(1,30):
    mi_di = torch.load(f"/users/nferruz/gboxo/Diffing Alpha Amylase New/M{i}_D{i}/diffing/checkpoint_latest.pt")["model_state_dict"]["W_dec"].detach().cpu()
    m0_di = torch.load(f"/users/nferruz/gboxo/Diffing Alpha Amylase New/M0_D{i}/diffing/checkpoint_latest.pt")["model_state_dict"]["W_dec"].detach().cpu()
    cs = compute_cs(base, mi_di)
    cs_m0 = compute_cs(base, m0_di)
    cs_list.append(cs)
    cs_m0_list.append(cs_m0)

# %%



i = 5 
x = cs_list[i].cpu().numpy()
y = cs_m0_list[i].cpu().numpy()
plt.scatter(x, y)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("M0_D0")
plt.ylabel("M0_D10")
plt.show()




# %%

x = np.load("/users/nferruz/gboxo/Diffing_Analysis_Data/firing_rates_M0_D1.npy")
x_log10 = np.log10(x + 1e-10)
# %%





















