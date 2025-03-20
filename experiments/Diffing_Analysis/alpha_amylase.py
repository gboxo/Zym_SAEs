import torch
import torch.nn.functional as F
import os
import re


"""
Things to compute:
    - For each type, each iteration CS with M0_D0
    - Iteration-wise CS
"""



def compute_CS(W_dec, W_dec_ref):
    cs = F.cosine_similarity(W_dec, W_dec_ref, dim=1)
    return cs

path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase New/"
files = os.listdir(path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

md_indices = [re.findall(r'\d+', file) for file in files]

sae_dict = {}

# Load M0_D0
string = "M0_D0"
sae_dict[string] = torch.load("/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/sae_training_iter_0_100/final/checkpoint_latest.pt", map_location=device)["model_state_dict"]["W_dec"].detach().cpu()

# Load FT
for data_index in range(1,7):
    string = f"M0_D{data_index}"
    sae_dict[string] = torch.load(path + string + "/diffing/checkpoint_latest.pt", map_location=device)["model_state_dict"]["W_dec"].detach().cpu()

# Load SAE
for index in range(1,7):
    string = f"M{index}_D{index}"
    sae_dict[string] = torch.load(path + string + "/diffing/checkpoint_latest.pt", map_location=device)["model_state_dict"]["W_dec"].detach().cpu()



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

os.makedirs("/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data", exist_ok=True)
torch.save(all_cs, "/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/all_cs.pt")



