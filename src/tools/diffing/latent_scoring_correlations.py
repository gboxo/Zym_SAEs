from diffing_utils import load_config
import pandas as pd
import argparse
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from plots_diffing import plot_correlation_heatmap, plot_firing_rate_vs_correlation, plot_2d_density, plot_3d_density

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

def obtain_features(df, output_dir):
    """
    Obtain features from natural sequences
    """
    sequences = df["sequence"].tolist()
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    pkl.dump(features, open(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
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



def firing_rates(features, output_dir):
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
    return firing_rates_seq








def get_correlations(mean_features, plddt, activity, tm_score, f_rates, cs, output_dir):
    """Calculate correlations between features and metrics."""
    # Calculate correlations between features and metrics
    correlations = []
    p_values = []
    
    for i in range(mean_features.shape[1]):
        feature = mean_features[:, i]
        if np.std(feature) > 0:
        
        # Calculate correlations with each metric
            corr_plddt, p_plddt = pearsonr(feature, plddt)
            corr_activity, p_activity = pearsonr(feature, activity)
            corr_tm, p_tm = pearsonr(feature, tm_score)
            
            # Store the correlations and p-values
            correlations.append([corr_plddt, corr_activity, corr_tm])
            p_values.append([p_plddt, p_activity, p_tm])
        else:
            correlations.append([0,0,0])
            p_values.append([0,0,0])
    
    correlations = np.array(correlations)
    p_values = np.array(p_values)
    
    # Calculate mean absolute correlation for each feature
    mean_abs_corr = np.mean(np.abs(correlations), axis=1)
    
    # Get the top features by mean absolute correlation
    top_k = 25
    top_indices = np.argsort(mean_abs_corr)[-top_k:][::-1]
    top_correlations = correlations[top_indices]
    top_p_values = p_values[top_indices]
    
    # Apply Benjamini-Hochberg correction for multiple testing
    mask = multipletests(p_values.flatten(), method='fdr_bh')[0].reshape(p_values.shape)
    
    # Create a correlation data dictionary
    correlation_data = {
        'feature_indices': top_indices,
        'correlations': correlations,
        'p_values': p_values,
        'significant': ~mask,
        'mean_abs_corr': mean_abs_corr,
        'f_rates': f_rates,
        'cs': cs
    }
    
    # Save the correlation data
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    pkl.dump(correlation_data, open(f"{output_dir}/correlations/top_correlations_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    
    print(f"Created correlation data with shape: {top_correlations.shape}")
    print(f"Number of significant correlations after correction: {np.sum(~mask)}")
    
    return correlation_data





def analyze_correlations(mean_features, plddt, activity, tm_score, f_rates, cs):
    """Main function to analyze correlations and create visualizations."""
    # Calculate correlations and get data
    correlation_data = get_correlations(mean_features, plddt, activity, tm_score, f_rates, cs)
    
    # Create various plots
    plot_correlation_heatmap(correlation_data, output_dir)
    plot_firing_rate_vs_correlation(correlation_data, output_dir)
    plot_2d_density(correlation_data['f_rates'], correlation_data['cs'], correlation_data['correlations'][:,1], output_dir)
    plot_3d_density(correlation_data, output_dir)
    
    return correlation_data


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)
    iteration_num = config["iteration_num"]
    model_iteration = config["model_iteration"]
    data_iteration = config["data_iteration"]
    ec_label = config["label"]
    cs_path = config["paths"]["cs_path"]
    df_path = config["paths"]["df_path"]
    output_dir = config["paths"]["output_dir"]
    model_path = config["paths"]["model_path"].format(iteration_num)
    sae_path = config["paths"]["sae_path"].format(model_iteration, data_iteration)

    cs = torch.load(cs_path)
    cs = cs[f"M{model_iteration}_D{data_iteration}_vs_M0_D0"].cpu().numpy()

    
    
    # Create the directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)



    
    

    assert os.path.exists(df_path), "Dataframe does not exist"
    df = pd.read_csv(df_path)
    if True:
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

        obtain_features(df, output_dir)

    features = load_features(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl")
    f_rates = firing_rates(features, output_dir)

    # Plot the histogram of the firing rates
    plt.hist(f_rates,bins=20)
    plt.savefig(f"{output_dir}/figures/firing_rates_histogram_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()



    mean_features = get_mean_features(features)[:,0]
    plddt = df["pLDDT"].tolist()
    plddt = np.array(plddt)

    activity = df["prediction1"].tolist()
    activity = np.array(activity)

    activity2 = df["prediction2"].tolist()
    activity2 = np.array(activity2)

    tm_score = df["alntmscore"].tolist()
    tm_score = np.array(tm_score)

    analyze_correlations(mean_features, plddt, activity, tm_score, f_rates, cs, output_dir)


