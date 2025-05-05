from src.tools.diffing.diffing_utils import load_config
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
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

"""
This is slightly modified version of the latent scoring script from the diffing repo, given that we are using a df from the DMS dataset
"""




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

def obtain_features(sequences, mutant, output_dir):
    """
    Obtain features from natural sequences
    """
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    features_dict = dict(zip(mutant, features))
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    pkl.dump(features_dict, open(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
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


def fit_lr_probe(X_train, y_train, X_test, y_test):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    """
    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-4.5, -3, 10)):
        lr_model = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1, max_iter=10000)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[1]
        print(active_features)
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


def get_important_features(X, pred, thresholds, directions='upper'):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    
    Args:
        X: Input features
        pred: Metrics to analyze
        thresholds: dict with keys 'pred' containing threshold values
        directions: str or dict, either 'upper' or 'lower' for all metrics, or dict with keys for each metric
    """
    # Standardize directions if string
    if isinstance(directions, str):
        directions = {k: directions for k in ['pred']}

    def get_mask(values, threshold, direction):
        return values > threshold if direction == 'upper' else values < threshold

    def get_assert(y_train, y_test):
        """
        Assert that y_train and y_test are have more than one class
        """
        assert len(np.unique(y_train)) > 1, "y_train must have more than one class"
        assert len(np.unique(y_test)) > 1, "y_test must have more than one class"

    def process_lr_probe(X, pred, thresholds, directions):
        """
        Process the LR probe
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, get_mask(pred, thresholds, directions))
        get_assert(y_train, y_test)
        results, probes = fit_lr_probe(X_train, y_train, X_test, y_test)
        return results, probes


    # Activity
    results_pred, probes_pred = process_lr_probe(X, pred, thresholds['pred'], directions['pred'])


    coefs_pred = torch.tensor(probes_pred[-1].coef_)[0]

    unique_coefs = torch.unique(torch.cat([torch.where(coefs_pred>0)[0]]))
    coefs = torch.stack([coefs_pred])
    coefs = coefs[:,unique_coefs]

    return unique_coefs, coefs







def analyze_important_features(mean_features, activity, thresholds, direction):
    """Main function to analyze important features and create visualizations."""
    # Calculate correlations and get data


    unique_coefs, coefs = get_important_features(mean_features, activity, thresholds, direction)
    
    # Create various plots
    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': coefs,
    }
    # Save the correlation data
    return importance_features
    

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
    df_path = config["paths"]["df_path"]
    output_dir = config["paths"]["out_dir"]
    model_path = config["paths"]["model_path"]
    sae_path = config["paths"]["sae_path"]





    
    
    # Create the directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)



    
    
    assert os.path.exists(df_path), "Dataframe does not exist"
    df = pd.read_csv(df_path)
    sequences = df["mutated_sequence"].tolist()
    activity = df["activity_dp7"].tolist()
    mutant = df["mutant"].tolist()
    
    activity = np.array(activity)
    activity_is_nan = np.isnan(activity)
    activity = activity[~activity_is_nan]
    mutant = np.array(mutant)
    mutant = [mutant[i] for i in range(len(mutant)) if not activity_is_nan[i]]
    sequences = [sequences[i] for i in range(len(sequences)) if not activity_is_nan[i]]



    if False:
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

        obtain_features(sequences, mutant, output_dir)

    dict_features = load_features(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl")
    activity = [df[df["mutant"] == key]["activity_dp7"].values[0] for key in dict_features.keys()]
    activity = np.array(activity)
    print(activity.shape)
    features = list(dict_features.values())
    print(len(features))


    f_rates = firing_rates(features, output_dir)

    # Plot the histogram of the firing rates
    plt.hist(f_rates,bins=20)
    plt.savefig(f"{output_dir}/figures/firing_rates_histogram_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()



    mean_features = get_mean_features(features)[:,0]

    


    def get_empirical_thresholds(activity):
        """
        For each value compute the 0.25 and 0.75 percentile and use it as the lower and upper threshold
        """
        activity_quantiles = np.percentile(activity, 95)
        thresholds_pos = {
            "pred": activity_quantiles,
        }
        return thresholds_pos



    
    # Get the empirical thresholds
    thresholds_pos = get_empirical_thresholds(activity)
    print(thresholds_pos)



    importance_features_pos = analyze_important_features(mean_features, activity, thresholds_pos, "upper")



    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    with open(f"{output_dir}/important_features/important_features_pos_M{model_iteration}_D{data_iteration}.pkl", "wb") as f:
        pkl.dump(importance_features_pos, f)
