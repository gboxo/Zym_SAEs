# %%
import pickle as pkl
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# PDB parser
from Bio.PDB import PDBParser

# %%
path = "/users/nferruz/gboxo/Diffing_Analysis_Data/"
features_path = os.path.join(path, "features","features_M2_D2.pkl")
with open(features_path, "rb") as f:
    features = pkl.load(f)

# %%

# Load data about the sequences
sequences_path = os.path.join(path, "dataframe_iteration2.csv")
sequences = pd.read_csv(sequences_path)

# %%

idx = 18
feats = features[idx]
feats = np.array(feats.todense())[:,714]
seq = sequences.loc[idx, "sequence"]
seq_id = sequences.loc[idx, "label"]
path = f"/users/nferruz/gboxo/Diffing_Analysis_Data/outputs/output_iteration2/PDB/{seq_id}.pdb"

parser = PDBParser()
structure = parser.get_structure("protein", path)
# I want to get the residues for each amino acid
# %%
non_active_residues = np.where(feats == 0)[0]
feats_norm = (feats - feats.min() + 1e-6) / (feats.max() - feats.min() + 1e-6)
feats_norm = feats_norm 
feats_norm[non_active_residues] = 0
residue_colors = {}
for model in structure:
    for chain in model:
        for i,residue in enumerate(chain):
            key = str((chain.id, residue.id[1]))
            residue_colors[key] = (
                float(1.0 if feats_norm[i] > 0.5 else 0.0),  # Red
                float(0.0),  # Green 
                float(1.0 if feats_norm[i] <= 0.5 and feats_norm[i] > 0.0 else 0.0)  # Blue
            )

import json
with open("/users/nferruz/gboxo/residue_colors.json", "w") as f:
    json.dump(residue_colors, f)


with open("/users/nferruz/gboxo/residue_colors.json", "r") as f:
    residue_colors = json.load(f)



# %%
