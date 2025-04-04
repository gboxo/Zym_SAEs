import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

base_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_25/sae_training_iter_0/final/checkpoint_latest.pt"
m0d4_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase Mixture 32/step_1c/M0_D4/diffing/checkpoint_latest.pt"
m4d4_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase Mixture 32/step_1c/M4_D4_stage2/diffing/checkpoint_latest.pt"

base_decoder = torch.load(base_path)["model_state_dict"]["W_dec"]
m0d4_decoder = torch.load(m0d4_path)["model_state_dict"]["W_dec"]
m4d4_decoder = torch.load(m4d4_path)["model_state_dict"]["W_dec"]

print(base_decoder.shape)
print(m0d4_decoder.shape)
print(m4d4_decoder.shape)

base_m0d4_cs = F.cosine_similarity(base_decoder, m0d4_decoder, dim=1).cpu().numpy()
base_m4d4_cs = F.cosine_similarity(base_decoder, m4d4_decoder, dim=1).cpu().numpy()

print(base_m0d4_cs)
print(base_m4d4_cs)

sns.scatterplot(x=base_m0d4_cs, y=base_m4d4_cs)
plt.savefig("base_m0d4_m4d4_cs_step_1c.png")
