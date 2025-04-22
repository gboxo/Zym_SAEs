import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from joypy import joyplot
all_cs = torch.load("/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/all_cs.pt")

cs_bm = []
for i in range(1,30):
    key = f"M0_D{i}_vs_M0_D0"
    cs = all_cs[key]
    cs_bm.append(cs)

cs_rl = []
for i in range(1,30):
    key = f"M{i}_D{i}_vs_M0_D0"
    cs = all_cs[key]-0.3
    cs_rl.append(cs)

data = []
for position in range(len(cs_bm)):
    bm = cs_bm[position]
    rl = cs_rl[position]
    for feat_id in range(len(bm)):
        data.append({
            'position': int(position),
            'feat_id': feat_id,
            'bm': bm[feat_id].numpy(),
            'rl': rl[feat_id].numpy()
        })

df = pd.DataFrame(data) 
df["position"] = df["position"].astype(int)
df["feat_id"] = df["feat_id"].astype(int)
df["bm"] = df["bm"].astype(float)
df["rl"] = df["rl"].astype(float)



ax, fig = joyplot(
    data=df[["bm", "rl", "position"]], 
    by="position",
    column=["bm", "rl"],
    color=["#F07605", "#9B1D20"],
    legend=True,
    alpha=0.5,
    figsize=(20, 16)
)
plt.title("Ridgeline Plot of BM and RL")
plt.savefig("ridge_plot.png")



