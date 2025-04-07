
import pandas as pd
import argparse
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from scipy.sparse import coo_matrix, vstack
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


def get_activations( model, tokenizer, sequence):
    sequence = "3.2.1.1<sep><start>" + sequence
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

def obtain_features(df):
    """
    Obtain features from natural sequences
    """
    sequences = df["sequence"].tolist()
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features", exist_ok=True)
    pkl.dump(features, open(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features/features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    del features
    torch.cuda.empty_cache()

def load_features(path):
    """
    Load features from a file
    """
    assert path.endswith(".pkl"), "File must end with .pkl"
    features = pkl.load(open(path, "rb"))
    return features

def get_mean_features(features):
    """
    Get the mean features
    """
    mean_features = []
    for feature in features:
        mean_features.append(feature.todense()[10:].sum(axis=0))
    mean_features = np.array(mean_features)
    return mean_features



def firing_rates(features):
    """
    Get the firing rates of the features

    1) Average number of firings per sequence with at least one firing
    2) Percentage of tokens that fire at least once per sequence
    3) Average number of firings per token
    """
    firing_rates_seq = []
    for feature in features:
        feats = feature.todense()[10:].sum(axis=0)
        w = np.where(feats > 0, 1, 0)
        fa = w.sum(axis=0)>0
        firing_rates_seq.append(fa)
    firing_rates_seq = np.array(firing_rates_seq).mean(axis=0)
    np.save(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/firing_rates_M{model_iteration}_D{data_iteration}.npy", firing_rates_seq)
    return firing_rates_seq





    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    # is_rl True if RL, False if SW
    parser.add_argument("--is_rl", action="store_true", help="Set to True if using RL, False if using SW")
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label
    ec_label = ec_label.strip()


    
    
    # Create the directories
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features", exist_ok=True)
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/firing_rates", exist_ok=True)
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/figures", exist_ok=True)




    
    

    if True:
        model_path = f"/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration{iteration_num}/" 
        sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase Mixture 32/last_iteration/M30_D30/diffing/"
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        # Load model
        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        torch.cuda.empty_cache()

        obtain_features(df)

    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features", exist_ok=True)
    features = load_features(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features/features_M30_D30.pkl")
    f_rates = firing_rates(features)

    # Plot the histogram of the firing rates
    plt.hist(f_rates,bins=20)
    plt.savefig(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/figures/firing_rates_histogram_M30_D30.png", dpi=300)
    plt.close()






