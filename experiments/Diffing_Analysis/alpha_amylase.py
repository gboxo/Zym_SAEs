import torch
import torch.nn.functional as F
import os
import re


"""
Things to compute:
    - For each type, each iteration CS with M0_D0
    - Iteration-wise CS
"""



path = f"/users/nferruz/gboxo/Diffing Alpha Amylase/"
files = os.listdir(path)

md_indices = [re.findall(r'\d+', file) for file in files]

# Load FT
sae_dict = {}
for data_index in range(4):
    string = f"M0_D{data_index}"
    sae_dict[string] = torch.load(path + string + "/diffing/checkpoint_latest.pt")["model_state_dict"]["W_dec"].detach().cpu()

# Load SAE
for index in range(4):
    string = f"M{index}_D{index}"
    sae_dict[string] = torch.load(path + string + "/diffing/checkpoint_latest.pt")["model_state_dict"]["W_dec"].detach().cpu()



def compute_CS(W_dec, W_dec_ref):
    cs = F.cosine_similarity(W_dec, W_dec_ref, dim=1)
    return cs

all_cs = {} 

# All vs M0_D0
for key in sae_dict.keys():
    if key == "M0_D0": continue
    cs = compute_CS(sae_dict[key], sae_dict["M0_D0"])
    all_cs[key+"_vs_M0_D0"] = cs

# Stage wise CS FT
for i in range(1,4):
    cs = compute_CS(sae_dict[f"M0_D{i}"], sae_dict[f"M0_D{i-1}"])
    all_cs[f"M0_D{i}_vs_M0_D{i-1}"] = cs

# Stage wise CS RL


for i in range(1,4):
    cs = compute_CS(sae_dict[f"M{i}_D{i}"], sae_dict[f"M{i-1}_D{i-1}"])
    all_cs[f"M{i}_D{i}_vs_M{i-1}_D{i-1}"] = cs

os.makedirs("Data/Diffing_Analysis_Data", exist_ok=True)
torch.save(all_cs, "Data/Diffing_Analysis_Data/all_cs.pt")



