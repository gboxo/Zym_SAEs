# %%
import pandas as pd
import sys
import os


files = os.listdir("/users/nferruz/gboxo/crg_boxo/Data/Diffing_Analysis_Data/ablation/M1_D1")
all_ids_seqs = {}
for file in files:
    with open(f"/users/nferruz/gboxo/crg_boxo/Data/Diffing_Analysis_Data/ablation/M1_D1/{file}", "r") as f:
        data = f.read()
        data = data.split("\n")
        data = [i for i in data if i != ""]
        # Get pairs of elements
        ids = [data[i] for i in range(len(data)) if i % 2 == 0]
        seqs = [data[i].strip("3 . 2 . 1 . 1 <sep> <start> ") for i in range(len(data)) if i % 2 == 1]
        ids_seqs = list(zip(ids, seqs))
        all_ids_seqs[file.split("_")[-1].strip(".txt")] = ids_seqs






# %%
import matplotlib.pyplot as plt
import seaborn as sns
average_length = []
for key, value in all_ids_seqs.items():
    lens = [len(i[1]) for i in value]
    average_length.append(sum(lens) / len(lens))

sns.kdeplot(average_length)
plt.show()


    # %%