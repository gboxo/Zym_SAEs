import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt





def fit_lr_probe(X_train, y_train):
    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-4.5, -3, 20)):
        lr_model = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[1]
        probes.append(lr_model)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        roc_auc = roc_auc_score(y_test, y_pred)
        results.append({
            "active_features": active_features,
            "sparsity": sparsity,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        })
    return results, probes



# ======== 
"""
For the features that predict high accuracy check in how many sequences they are present
"""

feature_path = "Data/Diffing_Analysis_Data/features_M2_D2.pkl"
df_path = "Data/Diffing_Analysis_Data/dataframe_iteration2.csv"
df = pd.read_csv(df_path)

with open(feature_path, "rb") as f:
    features = pkl.load(f)


# Get the mean of each activation
mean_features = [torch.tensor(feat.todense().sum(0)) for feat in features] 


X = torch.stack(mean_features)[:,0]
y = df["prediction1"].values 
pred1 = df["prediction1"].values 
pred2 = df["prediction2"].values 
plddt = df["pLDDT"].values 
tm_score = df["alntmscore"].values 

# Prediction 1
X_train, X_test, y_train, y_test = train_test_split(X,pred1>3)
results_pred1, probes_pred1 = fit_lr_probe(X_train,y_train)
# Prediction 2
X_train, X_test, y_train, y_test = train_test_split(X,pred2>1.5)
results_pred2, probes_pred2 = fit_lr_probe(X_train,y_train)
# PLDDT 
X_train, X_test, y_train, y_test = train_test_split(X,plddt>0.7)
results_plddt, probes_plddt = fit_lr_probe(X_train,y_train)
# TM-score
X_train, X_test, y_train, y_test = train_test_split(X,tm_score>0.8)
results_tm_score, probes_tm_score = fit_lr_probe(X_train,y_train)

coefs_pred1 = torch.tensor(probes_pred1[-1].coef_)[0]
coefs_pred2 = torch.tensor(probes_pred2[-1].coef_)[0]
coefs_plddt = torch.tensor(probes_plddt[-1].coef_)[0]
coefs_tm_score = torch.tensor(probes_tm_score[-1].coef_)[0]

unique_coefs = torch.unique(torch.cat([torch.where(coefs_pred1>0)[0],
                                       torch.where(coefs_pred2>0)[0],
                                       torch.where(coefs_plddt>0)[0],
                                       torch.where(coefs_tm_score>0)[0]]))
all_coefs = torch.stack([coefs_pred1, coefs_pred2, coefs_plddt, coefs_tm_score])
coefs = all_coefs[:,unique_coefs]
metrics = ["Prediction 1", "Prediction 2", "pLDDT", "TM-score"]
sns.heatmap(coefs)
plt.title("Coefficients of the features that predict high accuracy")
plt.xlabel("Feature index")
plt.ylabel("Model")
plt.xticks(range(len(unique_coefs)), unique_coefs.numpy().tolist(), rotation=90)
plt.yticks(range(len(metrics)), metrics, rotation=0)
plt.show()








all_coef = torch.tensor(probes[-1].coef_)[0]
idxs = results[-1]["active_features"]
idxs = torch.tensor(idxs)

coefs = all_coef[idxs]
positive_coefs_idxs = torch.where(coefs>0)
negative_coefs_idxs = torch.where(coefs<0)


activity_pos = X[:,idxs[positive_coefs_idxs]][idx_sort]
activity_neg = X[:,idxs[negative_coefs_idxs]][idx_sort]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(activity_pos>0, ax=ax1)
ax1.set_title("Positive coefficients")

sns.heatmap(activity_neg>0, ax=ax2) 
ax2.set_title("Negative coefficients")

plt.tight_layout()
plt.show()
