import pickle as pkl
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = "/media/workspace/features_M0_D1.pkl"


with open(path,"rb") as f:
    features = pkl.load(f)




"""
Questions that I want to answer:
    - For each feature in how many sequences it appears at least once
    - For each feature how much it's firing is correlated with the position
    - For each feature when it appears once how many times on average does it fire
    - For each feature that fires more thatn once in a sequence is the firing pattern sequential?

"""


# For features that fire at least once what is the mean number of fires per sequence
x = features[0].todense()
x = torch.tensor(x)

active_features = torch.where(x>0,1,0)
unique_active_features = active_features.sum(0)
active_features_indices = torch.where(unique_active_features>0)

number_of_fires = {}
for feat in active_features_indices[0]:
    number_of_fires[feat.item()] = unique_active_features[feat.item()]


plt.hist(unique_active_features.numpy(), bins=100)
plt.show()




# ======== COVERAGE (AT LEAST ONE FIRING) ================

running_count = torch.zeros(x.shape[1])
for seq in features:
    x = torch.tensor(seq.todense())
    active_features = torch.where(x>0,1,0)
    unique_active_features = active_features.sum(0) >0
    running_count += unique_active_features


plt.hist(running_count.numpy(), bins=100)
plt.show()


# ======== FIRING AND POSITION CORRELATION ================































