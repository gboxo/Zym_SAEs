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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

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


# New function to fit a single LR probe using CV
def fit_single_lr_probe(X_train, y_train, X_test, y_test, sparsity_values=None):
    """
    Fits a single sparse logistic regression probe using CV to find the best sparsity.

    Args:
        X_train, y_train: Training data and labels.
        X_test, y_test: Testing data and labels.
        sparsity_values: List of C values (inverse regularization strength) for CV.

    Returns:
        A tuple containing:
        - result (dict): Dictionary with evaluation metrics and model details.
        - lr_model (LogisticRegressionCV): The fitted model object.
    """
    if sparsity_values is None:
        # Default sparsity range similar to before
        sparsity_values = np.logspace(-4.5, -3, 10)

    # Use LogisticRegressionCV to find the best sparsity C
    lr_model = LogisticRegressionCV(
        cv=5,
        penalty="l1",
        solver="liblinear",
        class_weight="balanced",
        Cs=sparsity_values, # Cs is the inverse of regularization strength
        n_jobs=-1,
        scoring='roc_auc', # Use ROC AUC to select the best C
        random_state=42 # For reproducibility of CV splits
    )
    lr_model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred_proba = lr_model.predict_proba(X_test)[:, 1] # Probabilities for ROC AUC
    y_pred = lr_model.predict(X_test) # Predictions for accuracy

    accuracy = accuracy_score(y_test, y_pred)
    try:
        # roc_auc_score needs probabilities
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError as e:
         # Handle cases where only one class is predicted, or present in y_test
         print(f"  Warning: Could not calculate ROC AUC. {e}. Setting ROC AUC to 0.5.")
         roc_auc = 0.5


    coefs = lr_model.coef_[0] # Assuming binary classification
    active_features_indices = np.where(coefs != 0)[0]
    best_sparsity_param = lr_model.C_[0] # The best C found by CV

    result = {
        "active_features_indices": active_features_indices,
        "best_sparsity_param": best_sparsity_param,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        "coefs": coefs
    }

    return result, lr_model # Return the results dict and the fitted model


# Renamed and replaced get_important_features
def get_important_features_iterative(X, pred, plddt, tm_score, thresholds, directions='upper', max_iterations=10, min_roc_auc=0.55):
    """
    Iteratively fits sparse logistic regression probes to identify important features.
    In each iteration, features identified in previous iterations are zeroed out.
    Continues until performance drops below min_roc_auc or max_iterations is reached.
    
    Args:
        X: Input features (NumPy array).
        pred, plddt, tm_score: Metrics to predict (NumPy arrays).
        thresholds: dict with keys 'pred', 'plddt', 'tm_score' containing threshold values.
        directions: str or dict, either 'upper' or 'lower' for all metrics, or dict with keys for each metric.
        max_iterations: Maximum number of iterations per metric.
        min_roc_auc: The minimum ROC AUC score to continue iterating.

    Returns:
        A dictionary containing the results for each metric across iterations.
        Each metric's entry is a dict keyed by iteration number, storing coefs, roc_auc, etc.
        Example: {'pred': {1: {'coefs': ..., 'roc_auc': ...}, 2: {...}}, 'plddt': {...}, ...}
    """
    # Standardize directions if string
    if isinstance(directions, str):
        directions = {k: directions for k in ['pred', 'plddt', 'tm_score']}

    def get_mask(values, threshold, direction):
        # Return binary labels (0 or 1)
        return (values > threshold).astype(int) if direction == 'upper' else (values < threshold).astype(int)

    all_results = {'pred': {}, 'plddt': {}, 'tm_score': {}}
    metrics = {'pred': pred, 'plddt': plddt, 'tm_score': tm_score}
    original_X = X.copy() # Keep the original data

    for metric_name, metric_values in metrics.items():
        print(f"--- Processing Metric: {metric_name} ---")
        # Use the original X for each metric, zeroing out features cumulatively *within* that metric's iterations
        X_metric = original_X.copy() # Start fresh for each metric
        zeroed_features_indices = set() # Track features zeroed for this metric
        iteration = 1

        while iteration <= max_iterations:
            print(f"  Iteration {iteration}")

            # Create a modifiable copy for this iteration
            X_iter = X_metric.copy()

            # Zero out features found in previous iterations for this metric
            if zeroed_features_indices:
                # Convert set to list for indexing
                indices_to_zero = list(zeroed_features_indices)
                # Ensure indices are within bounds before zeroing
                valid_indices_to_zero = [idx for idx in indices_to_zero if idx < X_iter.shape[1]]
                if valid_indices_to_zero:
                    X_iter[:, valid_indices_to_zero] = 0
                # else:
                #     print(f"  Note: No valid feature indices to zero out for iteration {iteration}.")


            # Prepare data for this iteration
            y = get_mask(metric_values, thresholds[metric_name], directions[metric_name])

            # Check for sufficient data variability before splitting and fitting
            if len(np.unique(y)) < 2:
                 print(f"  Stopping: Only one class present for metric '{metric_name}' with threshold {thresholds[metric_name]} ({directions[metric_name]}). Cannot train classifier.")
                 break # Stop iterations for this metric

            # Split data - use stratification and iteration-dependent random state
            X_train, X_test, y_train, y_test = train_test_split(
                X_iter, y, test_size=0.3, random_state=iteration, stratify=y)

            # Check class balance after split
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                 print(f"  Warning: Iteration {iteration} for metric '{metric_name}' resulted in train/test split with only one class (Train: {np.unique(y_train)}, Test: {np.unique(y_test)}).")
                 print("  Stopping iterations for this metric due to split issue.")
                 break


            # Fit the single LR probe using CV
            try:
                 # Use the new function
                 iter_result, _ = fit_single_lr_probe(X_train, y_train, X_test, y_test)
                 iteration_roc_auc = iter_result['roc_auc']
                 iteration_coefs = iter_result['coefs']

                 print(f"  Iteration {iteration} ROC AUC: {iteration_roc_auc:.4f}")

                 # Check stopping condition (performance) - don't stop on iter 1 just because ROC is low
                 if iteration_roc_auc < min_roc_auc and iteration > 1:
                     print(f"  Stopping: ROC AUC ({iteration_roc_auc:.4f}) below threshold ({min_roc_auc}) after iteration 1.")
                     break

                 # Identify *all* non-zero features in this iteration's model
                 # Indices are relative to the original feature space
                 current_non_zero_indices = set(iter_result['active_features_indices'])

                 # Check stopping condition (no new features found, considering those already zeroed)
                 newly_found_indices = current_non_zero_indices - zeroed_features_indices
                 if not newly_found_indices and iteration > 1 : # Don't stop on iter 1 if no features found initially
                     print(f"  Stopping: No *new* non-zero features found in iteration {iteration}.")
                     break

                 # Store results for this iteration (store the whole dictionary)
                 all_results[metric_name][iteration] = iter_result


                 # Update the set of features to zero out for the *next* iteration
                 # Add all non-zero features found in *this* iteration
                 zeroed_features_indices.update(current_non_zero_indices)


            except Exception as e:
                 # Catch potential errors during model fitting (e.g., convergence issues, data errors)
                 print(f"  Error during fitting/evaluation for iteration {iteration}, metric '{metric_name}': {e}")
                 print("  Stopping iterations for this metric due to error.")
                 break # Stop iterations for this metric

            iteration += 1

        if iteration > max_iterations:
             print(f"  Stopping: Reached max iterations ({max_iterations}).")


    return all_results

    

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
    df_path = config["paths"]["df_path"].format(iteration_num=iteration_num)
    output_dir = config["paths"]["out_dir"]
    model_path = config["paths"]["model_path"]
    sae_path = config["paths"]["sae_path"]
    disc_thresholds = config["thresholds"]




    cs = torch.load(cs_path)
    cs = cs[f"M{model_iteration}_D{data_iteration}_vs_M0_D0"].cpu().numpy()

    
    
    # Create the directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    os.makedirs(f"{output_dir}/iterative_important_features", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)



    
    

    assert os.path.exists(df_path), "Dataframe does not exist"
    df = pd.read_csv(df_path)

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

        obtain_features(df, output_dir)

    features = load_features(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl")
    f_rates = firing_rates(features, output_dir)

    # Plot the histogram of the firing rates
    plt.hist(f_rates,bins=20)
    plt.savefig(f"{output_dir}/figures/firing_rates_histogram_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()



    mean_features = get_mean_features(features)[:,0]
    # Ensure mean_features is 2D for Logistic Regression (n_samples, n_features)
    if mean_features.ndim == 1:
        mean_features = mean_features.reshape(-1, 1)

    plddt = df["pLDDT"].tolist()
    plddt = np.array(plddt)

    activity = df["prediction"].tolist()
    activity = np.array(activity)


    tm_score = df["alntmscore"].tolist()
    tm_score = np.array(tm_score)

    def get_thresholds_and_directions(thresholds):
        thresholds_pos = {}
        thresholds_neg = {}
        for key, value in thresholds.items():
            thresholds_pos[key] = value["upper"]
            thresholds_neg[key] = value["lower"]
        return thresholds_pos, thresholds_neg

    def get_empirical_thresholds(activity, plddt, tm_score):
        """
        For each value compute the 0.25 and 0.75 percentile and use it as the lower and upper threshold
        """
        activity_quantiles = np.percentile(activity, [25, 75])
        plddt_quantiles = np.percentile(plddt, [25, 75])
        tm_score_quantiles = np.percentile(tm_score, [25, 75])
        thresholds_pos = {
            "pred": activity_quantiles[0],
            "plddt": plddt_quantiles[0],
            "tm_score": tm_score_quantiles[0],
        }
        thresholds_neg = {
            "pred": activity_quantiles[1],
            "plddt": plddt_quantiles[1],
            "tm_score": tm_score_quantiles[1],
        }
        return thresholds_pos, thresholds_neg



    
    # Get the empirical thresholds
    thresholds_pos, thresholds_neg = get_empirical_thresholds(activity, plddt, tm_score)


    # --- Updated Analysis Section ---
    print("\nRunning iterative feature analysis for Positive Correlation (Upper Thresholds)...")
    iterative_results_pos = get_important_features_iterative(
        mean_features, activity, plddt, tm_score, thresholds_pos, directions="upper",
        max_iterations=10, min_roc_auc=0.55 # Example parameters, adjust as needed
    )

    print("\nRunning iterative feature analysis for Negative Correlation (Lower Thresholds)...")
    iterative_results_neg = get_important_features_iterative(
        mean_features, activity, plddt, tm_score, thresholds_neg, directions="lower",
        max_iterations=10, min_roc_auc=0.55 # Example parameters, adjust as needed
    )

    # Save the iterative results
    output_subdir = f"{output_dir}/important_features_iterative"
    os.makedirs(output_subdir, exist_ok=True)
    pos_results_path = f"{output_subdir}/iterative_features_pos_M{model_iteration}_D{data_iteration}.pkl"
    neg_results_path = f"{output_subdir}/iterative_features_neg_M{model_iteration}_D{data_iteration}.pkl"

    print(f"\nSaving positive correlation iterative results to {pos_results_path}")
    with open(pos_results_path, "wb") as f:
        pkl.dump(iterative_results_pos, f)

    print(f"Saving negative correlation iterative results to {neg_results_path}")
    with open(neg_results_path, "wb") as f:
        pkl.dump(iterative_results_neg, f)

    print("\nIterative analysis complete.")
