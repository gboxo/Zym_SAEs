import pandas as pd
import pickle as pkl
import os

import numpy as np

from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression



exp_df = "/home/woody/b114cb/b114cb23/boxo/experimental_SAPI.csv"
exp_df = pd.read_csv(exp_df)
dms_df = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
dms_df = pd.read_csv(dms_df)

exp_embs_dict = pkl.load(open("/home/woody/b114cb/b114cb23/boxo/esm_lr/embs_dict_exp_SAPI.pkl", "rb"))
dms_embs_dict = pkl.load(open("/home/woody/b114cb/b114cb23/boxo/esm_lr/embs_dict.pkl", "rb"))



X_exp = np.array([exp_embs_dict[i] for i in exp_embs_dict.keys()])
y_exp = [exp_df[exp_df["Unnamed: 0"] == i]["SAPI"].values[0] for i in exp_embs_dict.keys()]

X_dms = np.array([dms_embs_dict[i] for i in dms_embs_dict.keys()])
y_dms = [dms_df[dms_df["mutant"] == mutant]["activity_dp7"].values[0] for mutant in dms_embs_dict.keys()]




X = np.concatenate([X_exp, X_dms], axis=0)
y = np.concatenate([np.array(y_exp), np.array(y_dms)], axis=0)


is_na = np.isnan(y)
X = X[~is_na]
y = y[~is_na]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print(pearsonr(y_pred, y_test))


with open("/home/woody/b114cb/b114cb23/boxo/esm_lr/lr_exp_dms.pkl", "wb") as f:
    pkl.dump(lr, f)






