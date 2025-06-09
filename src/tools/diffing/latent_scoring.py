from src.tools.diffing.diffing_utils import load_config
import pandas as pd
import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from src.tools.diffing.feature_extraction import remove_nan_values
from src.tools.diffing.latent_scoring_config import LatentScoringConfig

"""
This is slightly modified version of the latent scoring script from the diffing repo, given that we are using a df from the DMS dataset
"""


def get_mean_features(features, prefix_tokens=10):
    """
    Get the mean features
    """
    mean_features = []
    for feature in features:
        mean_features.append(feature.todense()[prefix_tokens:].sum(axis=0)) # Hardcoded to skip the first 10 tokens, which are the prompt 
    mean_features = np.array(mean_features)
    return mean_features

def load_features(path):
    """
    Load features from a file
    """
    assert path.endswith(".pkl"), "File must end with .pkl"
    features = pkl.load(open(path, "rb"))

    return features



def fit_lr_probe(X_train, y_train, X_test, y_test):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    """
    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-3.5, -2, 10)):
        lr_model = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1, max_iter=10000)
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


    # Add up the coefficients of all the probes and mask all but the top 10 coefficients
    coefs_pred_full = torch.stack([torch.tensor(probe.coef_).squeeze() for probe in probes_pred]).sum(dim=0)
    topk = torch.topk(coefs_pred_full.abs(), 10)
    mask = torch.zeros_like(coefs_pred_full, dtype=torch.bool)
    mask[topk.indices] = True
    coefs_pred = coefs_pred_full * mask
    


    unique_coefs = torch.unique(torch.cat([torch.where(coefs_pred!=0)[0]]))
    coefs = torch.stack([coefs_pred])
    coefs = coefs[:,unique_coefs]

    return unique_coefs, coefs



def get_features_greater_than_min_activity(X_train, y_train, min_activity=2, min_rest_fraction=0.01):

    """
    Get the features that don't fire for sequences with activty higher than min_activity,
    but do fire on at least min_rest_fraction of the remaining sequences.
    
    Args:
        min_activity: Minimum activity value for sequences to be considered
        min_rest_fraction: Minimum fraction of remaining sequences that must have the feature (default 0.1)
    
    Returns:
        Array of indices for features that meet the criteria
    """


    def get_mask(values, threshold):
        return values > threshold

    def get_assert(y_train):
        """
        Assert that y_train is have more than one class
        """
        assert len(np.unique(y_train)) > 1, "y_train must have more than one class"

    def process_features(pred, thresholds):
        """
        Process the LR probe
        """
        y_train = get_mask(pred, thresholds)
        get_assert(y_train)
        return y_train

    top_mask = process_features(activity, min_activity)
    # Split into top x% and rest
    rest_mask = ~top_mask
    

    # Get features for each group
    top_features = X_train[top_mask]
    rest_features = X_train[rest_mask]
    
    # Find features that don't fire in top x%
    top_zero = np.all(top_features == 0, axis=0)
    
    # Find features that fire in at least min_rest_fraction of rest
    rest_firing = np.sum(rest_features > 0, axis=0) / rest_features.shape[0]
    rest_sufficient = rest_firing >= min_rest_fraction
    
    # Return indices where both conditions are met
    unique_coefs = np.where(top_zero & rest_sufficient)[0]

    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': None,
    }

    return importance_features
    
    




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


    df_path = config["paths"]["df_path"]
    output_dir = config["paths"]["out_dir"]
    model_path = config["paths"]["model_path"]
    sae_path = config["paths"]["sae_path"]

    seq_col_id = config["df"]["seq_col_id"]
    pred_col_id = config["df"]["pred_col_id"]
    col_id = config["df"]["col_id"]

    model_name = config["model_name"]
    ec_label = config["label"]
    DMS = config["is_DMS"]
    hook_point = config["hook_point"]
    
    prefix_tokens = config.get("prefix_tokens", 9)  # Default to 10 if not specified
    percentiles = config.get("percentiles", [94, 96, 98])  # Default to [94, 96, 98] if not specified
    min_rest_fractions = config.get("min_rest_fraction", [0.05,0.1])  # Default to 0.01 if not specified

    latent_scoring_config = LatentScoringConfig(
            hook_point=hook_point,
            model_name=model_name,
            is_DMS=DMS,
            label=ec_label,
            df_path=df_path,
            model_path=model_path,
            sae_path=sae_path,
            out_dir=output_dir,
            seq_col_id=seq_col_id,
            pred_col_id=pred_col_id,
            col_id=col_id,
            prefix_tokens=prefix_tokens,
            percentiles=percentiles,
            min_rest_fraction=min_rest_fractions
            )

    

    df = pd.read_csv(df_path)

    print(df.head())
    print(df.columns)

    sequences = df[seq_col_id].tolist()
    activity = df[pred_col_id].tolist()
    mutant = df[col_id].tolist()






    
    dict_features = load_features(f"{output_dir}/features/features_{model_name}.pkl")
    if DMS:
        sequences, activity, mutant = remove_nan_values(sequences, activity, mutant)


    activity = [df[df[col_id] == key][pred_col_id].values[0] for key in dict_features.keys()]

    features = list(dict_features.values())


    # DEFINE THE PERCENTILE TO USE

    mean_features = get_mean_features(features, prefix_tokens)[:,0]

    def get_empirical_thresholds(activity,pth):
        """
        Get empirical thresholds based on percentiles of activity.
        """
        activity_quantiles = np.percentile(activity, pth)
        thresholds_pos = {
            "pred": activity_quantiles,
        }
        return thresholds_pos



    
    # Get the empirical thresholds

    for pth in percentiles:
        thresholds_pos = get_empirical_thresholds(activity,pth)

        # Get the features that are greater than the threshold
        for min_rest_fraction in min_rest_fractions:
            features_greater_than_threshold = get_features_greater_than_min_activity(mean_features, activity, min_activity=thresholds_pos["pred"], min_rest_fraction=min_rest_fraction)
            if len(features_greater_than_threshold["unique_coefs"]) == 0:
                continue

            with open(f"{output_dir}/important_features/important_features_{model_name}_{pth}_{min_rest_fraction}_ablation.pkl", "wb") as f:
                pkl.dump(features_greater_than_threshold, f)

        importance_features_pos = analyze_important_features(mean_features, activity, thresholds_pos, "upper")
        os.makedirs(f"{output_dir}/important_features", exist_ok=True)
        
        with open(f"{output_dir}/important_features/important_features{model_name}_{pth}.pkl", "wb") as f:
            pkl.dump(importance_features_pos, f)


