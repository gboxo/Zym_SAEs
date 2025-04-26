
import pandas as pd
import pickle as pkl
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LassoCV
from scipy.sparse import coo_matrix, vstack
from tqdm import tqdm
from prettytable import PrettyTable
# %%




def get_data():
    # The first line is the header
    data = pd.read_csv("/users/nferruz/gboxo/sae_oracle/alpha-amylase-training-data.csv",header=0)

    # The first column is the sequence
    return data

def get_activations( model, tokenizer, sequence):
    sequence = "3.2.1.1<sep><start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("25.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.25.hook_resid_pre"]
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

def get_max_features(features):
    return np.max(features, axis=0)

def obtain_features(df, features):
    features = get_max_features(features)


    features = [features[i] for i in range(len(features))]

    labels = [df["activity_dp7"][i] for i in range(len(df))]

    features = vstack(features)
    labels = np.array(labels)
    os.makedirs(f"/users/nferruz/gboxo/sae_oracle/features", exist_ok=True)
    
    # Save everything using pickle instead of mixed formats
    with open(f"/users/nferruz/gboxo/sae_oracle/features/features.pkl", "wb") as f:
        pkl.dump(features, f)
    with open(f"/users/nferruz/gboxo/sae_oracle/features/labels.pkl", "wb") as f:
        pkl.dump(labels, f)
        
        
    del features, labels
    torch.cuda.empty_cache()


def load_features():
    # Update loading to use pickle for all files
    with open(f"/users/nferruz/gboxo/sae_oracle/features/features.pkl", "rb") as f:
        features = pkl.load(f)
    with open(f"/users/nferruz/gboxo/sae_oracle/features/labels.pkl", "rb") as f:
        labels = pkl.load(f)


    # No need for additional processing since we're loading directly from pickle
    return features, labels


    
    
    


def main(model, jump_relu, tokenizer):
    df = get_data()
    sequences = df["mutated_sequence"].values
    features = get_all_features(model, jump_relu, tokenizer, sequences)
    obtain_features(df, features)
    train_features, test_features, train_labels, test_labels = load_features()
    probes, results = train_linear_probe(train_features, test_features, train_labels, test_labels)
    train_table = display_training_results(results)
    print(train_table)
    test_results = test_linear_probe(probes, test_features, test_labels)
    test_table = display_testing_results(test_results)
    print(test_table)
    
    # Save results to file
    save_results_to_file(train_table, test_table)

if __name__ == "__main__":

    if True:
        model_path = "AI4PD/ZymCTRL"
        sae_path="/users/nferruz/gboxo/SAE_2025_04_02_32_15360_25/sae_training_iter_0/final/"


        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        torch.cuda.empty_cache()

    main(model, jump_relu, tokenizer)
