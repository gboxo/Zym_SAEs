
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd


path = "/home/woody/b114cb/b114cb23/boxo/dataset_analysis_sapi_only/pairwise_alignment_results"
files = os.listdir(path)


all_dataframes = []
for file in files :
    n = file.split("_")[-1].split(".")[0].replace("iteration", "")
    n = int(n)
    df = pd.read_csv(os.path.join(path, file), sep = "\t", header = None)
    df.columns = ["query", "target", "alntmscore", "alnlen", "qstart", "qend","_unk_", "tstart", "tend", "evalue", "bits", "cigar"]
    df["iteration"] = n
    all_dataframes.append(df)




df = pd.concat(all_dataframes)
df.sort_values(by = "iteration", inplace = True)
print(df.head())


# Violin plot of the distribution of tm scores in each iteration
plt.figure(figsize=(15, 6))
sns.violinplot(data=df, x="iteration", y="alntmscore")
plt.xlabel("Iteration")
plt.ylabel("TM Score")
plt.title("Violin plot of the distribution of tm scores in each iteration")
plt.savefig("/home/woody/b114cb/b114cb23/boxo/dataset_analysis_sapi_only/tm_score_distribution.png")
plt.close()


