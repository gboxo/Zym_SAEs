import torch
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, RegressorMixin



embs_path  = "/home/woody/b114cb/b114cb23/ZF_FT_alphaamylase_gerard/predictor_test/emb_out"
df_path = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
out_dir = "/home/woody/b114cb/b114cb23/boxo/esm_lr"
os.makedirs(out_dir, exist_ok=True)

files = os.listdir(embs_path)
files = [f for f in files if f.endswith(".pt")]


df = pd.read_csv(df_path)

embs_dict = {}
for file in files:
    elem = torch.load(os.path.join(embs_path, file))
    key = elem["label"]
    embs_dict[key] = elem["mean_representations"][33]

#import pickle as pkl
#with open(os.path.join(out_dir, "embs_dict.pkl"), "rb") as f:
#    embs_dict = pkl.load(f)


X = []
y = []

for mutant in df["mutant"].unique():
    activity = df[df["mutant"] == mutant]["activity_dp7"].values
    if not np.isnan(activity[0]):
        X.append(embs_dict[mutant])
        y.append(activity[0].item())

X = torch.stack(X)
y = torch.tensor(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== Train a Simple Linear Regression Model ====

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pearson_corr = pearsonr(y_pred, y_test)

print(f"Pearson Correlation: {pearson_corr}")

df_pred = pd.DataFrame({"pred": y_pred, "true": y_test})

print(df_pred.describe(percentiles=[0.01,0.25, 0.5, 0.75, 0.95, 0.99]))


# ======= Fit a MLP Regressor =======

from sklearn.neural_network import MLPRegressor

model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=10000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
pearson_corr = pearsonr(y_pred, y_test)

print(f"Pearson Correlation: {pearson_corr}")

df_pred = pd.DataFrame({"pred": y_pred, "true": y_test})

print(df_pred.describe(percentiles=[0.01,0.25, 0.5, 0.75, 0.95, 0.99]))

import pickle as pkl
with open(os.path.join(out_dir, "mlp_regressor.pkl"), "wb") as f:
    pkl.dump(model, f)

