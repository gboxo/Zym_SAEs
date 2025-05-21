import pandas as pd
import pickle as pkl
import numpy as np
import os


path = "/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/important_features/"
files = os.listdir(path)

files = [f for f in files if f.endswith(".pkl")]
files = [f for f in files if "ablation" not in f]



for file in files:
    print(file)
    with open(path+file, "rb") as f:
        data = pkl.load(f)
    print(data)





file_path = path + "/important_features_pos_M0_D0_97.pkl"
with open(file_path, "rb") as f:
    important_features = pkl.load(f)

coefs = important_features["unique_coefs"]


features_path = "/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/features/features_M0_D0.pkl"
with open(features_path, "rb") as f:
    features = pkl.load(f)


key = list(features.keys())[0]
eg = np.array(features[key].todense())

max_activations = np.zeros(eg.shape[1])
for key in features.keys():
    max_activations = np.maximum(max_activations, np.array(features[key].todense()).max(axis=0))

with open(path + "/max_features_M0_D0.pkl", "wb") as f:
    pkl.dump(max_activations, f)







