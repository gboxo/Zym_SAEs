
import pandas as pd

import pickle as pkl
import json
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.cluster.hierarchy as hierarchy
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, vstack
from prettytable import PrettyTable
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from src.utils import get_paths
# %%


def collect_data():
    alpha_path = "/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_TM_aln/alpha_4.2.1.1_TM_iteration0"
    beta_path = "/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_TM_aln/beta_4.2.1.1_TM_iteration0"
    gen_path = "/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_TM_aln/seq_gen_4.2.1.1_iteration0.fasta"

    with open(alpha_path, "r") as f:
        alpha = f.read()
        alpha = alpha.split("\n")
        alpha = [elem.split("\t") for elem in alpha]
        alpha = pd.DataFrame(alpha)
        alpha.index = alpha[0]
        alpha = alpha.drop(0, axis=1)

    with open(beta_path, "r") as f:
        beta = f.read()
        beta = beta.split("\n")
        beta = [elem.split("\t") for elem in beta]
        beta = pd.DataFrame(beta)
        beta.index = beta[0]
        beta = beta.drop(0, axis=1)

    with open(gen_path, "r") as f:
        gen = f.read()
        gen = gen.split(">")
        gen = [g for g in gen if g != ""]
        ec = [g.split("\n")[0].split("\t")[0] for g in gen]
        score = [g.split("\n")[0].split("\t")[1] for g in gen]
        seq = [g.split("\n")[1] for g in gen]
        gen = pd.DataFrame({"ec": ec, "seq": seq, "score": score})
    
    indices = gen["ec"].values

    alpha_values = alpha.loc[indices][2]
    beta_values = beta.loc[indices][2]

    joined = pd.DataFrame({ "alpha": alpha_values, "beta": beta_values, "seq": gen["seq"].values})
    joined["class"] = joined["alpha"] > joined["beta"]
    joined["class"] = joined["class"].astype(int)
    joined.to_csv("alpha_beta_class.csv", index=False)

def get_data():
    data = pd.read_csv("alpha_beta_class.csv")
    return data

def get_activations( model, tokenizer, sequence):
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("26.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.26.hook_resid_pre"]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]
    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def get_all_features(model, sae, tokenizer, sequences):
    all_features = []
    for sequence in tqdm(sequences):
        activations = get_activations(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features)
        del activations, features
        torch.cuda.empty_cache()
    return all_features

def obtain_features(df,features):

    random_indices = np.random.permutation(len(features))
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    test_indices = random_indices[int(len(random_indices)*0.8):]

    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]
    train_labels = [[df["class"][i]]*features[i].shape[0] for i in train_indices]
    test_labels = [[df["class"][i]]*features[i].shape[0] for i in test_indices]

    train_features = vstack(train_features)
    os.makedirs(f"Data/Alpha_Beta_Data/features", exist_ok=True)
    np.savez(f"Data/Alpha_Beta_Data/features/features_train.npz",train_features)
    np.save(f"Data/Alpha_Beta_Data/features/features_test.npy",test_features)

    with open(f"Data/Alpha_Beta_Data/features/labels_train.pkl", "wb") as f:
        pkl.dump(train_labels, f)
    with open(f"Data/Alpha_Beta_Data/features/labels_test.pkl", "wb") as f:
        pkl.dump(test_labels, f)
    del features, train_features, test_features, random_indices, train_indices, test_indices
    torch.cuda.empty_cache()


def load_features():
    train_features = np.load(f"Data/Alpha_Beta_Data/features/features_train.npz", allow_pickle=True)
    test_features = np.load(f"Data/Alpha_Beta_Data/features/features_test.npy", allow_pickle=True)
    with open(f"Data/Alpha_Beta_Data/features/labels_train.pkl", "rb") as f:
        train_labels = pkl.load(f)
    with open(f"Data/Alpha_Beta_Data/features/labels_test.pkl", "rb") as f:
        test_labels = pkl.load(f)

    train_features = train_features["arr_0"].tolist()
    test_features = test_features.tolist()
    test_features = vstack(test_features)
    train_labels = train_labels
    test_labels = test_labels

    return train_features, test_features, train_labels, test_labels



def train_linear_probe(train_natural_features, train_synth_features, test_natural_features, test_synth_features):
    # Concatennate  COO

    X_train = vstack((train_natural_features, train_synth_features))
    X_test = vstack((vstack(test_natural_features), vstack(test_synth_features)))

    y_train = np.concatenate((np.zeros(train_natural_features.shape[0]), np.ones(train_synth_features.shape[0])), axis=0)
    y_test = np.concatenate((np.zeros(vstack(test_natural_features).shape[0]), np.ones(vstack(test_synth_features).shape[0])), axis=0)
    

    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-5, -3, 20)):
        lr_model = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[1]
        probes.append(lr_model)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        roc_auc = roc_auc_score(y_test, y_pred)
        results.append({
            "active_features": len(active_features),
            "sparsity": sparsity,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        })
    best_result = max(results, key=lambda x: x["roc_auc"])
    return probes, results

def test_linear_probe(probes, test_natural_features, test_synth_features, threshold=0.5):
    """
    Implement a voting scheme to get the final prediction using an ensemble of probes
    
    Args:
        probes: List of trained logistic regression probes
        test_natural_features: Features from natural samples
        test_synth_features: Features from synthetic samples
        threshold: Classification threshold (default 0.5, will be tuned for 1% FPR)
        
    Returns:
        Dictionary containing predictions and probabilities for each class
    """

    # Store predictions from each probe
    results = []
    label_dict = {"alpha": 0, "beta": 1}
    
    for probe in probes:
        all_predictions = []
        all_probs = []
        true_labels = []
        for label, test_features in [("alpha", test_natural_features), ("beta", test_synth_features)]:
            for test_feature in test_features:
                # Get predictions from single probe
                true_labels.append(label_dict[label])
                # Get probability estimates
                probs = probe.predict_proba(test_feature)
                all_probs.append(probs[:, 1])  # Probability of synthetic class
                
                # Get binary predictions
                pred = (probs[:, 1] >= threshold).astype(int)

                all_predictions.append(pred)
        
        final_probs = np.array([prob.mean() for prob in all_probs])
        final_preds = (final_probs >= threshold).astype(int)
        accuracy, precision, recall, f1, roc_auc = compute_metrics(final_probs, final_preds, true_labels)
        results.append({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc
        })
        
    return results

def compute_metrics(probabilities, predictions, true_labels):
    accuracy = accuracy_score(predictions, true_labels)
    precision = precision_score(predictions, true_labels)
    recall = recall_score(predictions, true_labels)
    f1 = f1_score(predictions, true_labels)
    roc_auc = roc_auc_score(true_labels, probabilities)
    return accuracy, precision, recall, f1, roc_auc

def main():
    df = get_data()
    sequences = df["seq"].values
    #features = get_all_features(model, jump_relu, tokenizer, sequences)
    #obtain_features(df, features)
    train_features, test_features, train_labels, test_labels = load_features()
    probes, results = train_linear_probe(train_features, test_features, train_labels, test_labels)
    test_linear_probe(probes, test_features, test_labels)

if __name__ == "__main__":

    if True:
        paths = get_paths()
        model_path = paths.model_path
        sae_path = paths.sae_path

        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        torch.cuda.empty_cache()
    main()