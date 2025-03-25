# %%
import pickle as pkl
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# %%
path = "/users/nferruz/gboxo/Diffing_Analysis_Data/"
features_path = os.path.join(path, "features","features_M4_D4.pkl")
with open(features_path, "rb") as f:
    features = pkl.load(f)

# %%

# Load data about the sequences
sequences_path = os.path.join(path, "dataframe_iteration4.csv")
sequences = pd.read_csv(sequences_path)

# %%

seq_id = 939
feats = features[seq_id]
seq = sequences.loc[seq_id, "sequence"]

# %%







