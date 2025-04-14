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
    data = pd.read_csv("/home/woody/b114cb/b114cb23/boxo/sae_oracle/alpha-amylase-training-data.csv",header=0)

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

    random_indices = np.random.permutation(len(features))
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    test_indices = random_indices[int(len(random_indices)*0.8):]

    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]

    train_labels = [df["activity_dp7"][i] for i in train_indices]
    test_labels = [df["activity_dp7"][i] for i in test_indices]

    train_features = vstack(train_features)
    test_features = vstack(test_features)
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features", exist_ok=True)
    
    # Save everything using pickle instead of mixed formats
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/features_train.pkl", "wb") as f:
        pkl.dump(train_features, f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/features_test.pkl", "wb") as f:
        pkl.dump(test_features, f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/labels_train.pkl", "wb") as f:
        pkl.dump(train_labels, f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/labels_test.pkl", "wb") as f:
        pkl.dump(test_labels, f)
        
    del features, train_features, test_features, random_indices, train_indices, test_indices
    torch.cuda.empty_cache()


def load_features():
    # Update loading to use pickle for all files
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/features_train.pkl", "rb") as f:
        train_features = pkl.load(f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/features_test.pkl", "rb") as f:
        test_features = pkl.load(f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/labels_train.pkl", "rb") as f:
        train_labels = pkl.load(f)
    with open(f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/features/labels_test.pkl", "rb") as f:
        test_labels = pkl.load(f)

    # No need for additional processing since we're loading directly from pickle
    return train_features, test_features, train_labels, test_labels



def train_linear_probe(train_features, test_features, train_labels, test_labels):
    # Concatenate COO
    X_train = train_features
    X_test = test_features

    y_train = np.concatenate(train_labels)
    y_test = np.concatenate(test_labels)

    results = []
    probes = []
    
    # Using LassoCV instead of LogisticRegressionCV for linear regression with L1 penalty
    for sparsity in tqdm(np.logspace(-4.5, -3, 20)):
        # Convert sparsity to alpha parameter for Lasso
        alpha = sparsity
        lasso_model = LassoCV(cv=5, alphas=[alpha], max_iter=10000, n_jobs=-1)
        lasso_model.fit(X_train, y_train)
        
        coefs = lasso_model.coef_.reshape(1, -1)  # Reshape to match logistic regression format
        active_features = np.where(coefs != 0)[1]
        probes.append(lasso_model)
        
        y_pred = lasso_model.predict(X_test)
        
        # For regression metrics, we'll use R² instead of accuracy/ROC-AUC
        r2_score = lasso_model.score(X_test, y_test)
        mse = np.mean((y_test - y_pred)**2)
        
        results.append({
            "active_features": active_features,
            "sparsity": sparsity,
            "r2_score": r2_score,
            "mse": mse,
        })
    
    # Select best model based on R² score
    best_result = max(results, key=lambda x: x["r2_score"])
    return probes, results

def test_linear_probe(probes, test_features, test_labels, threshold=None):
    """
    Test linear regression probes on test data
    
    Args:
        probes: List of trained Lasso regression probes
        test_features: Features from test samples
        test_labels: Labels from test samples
        threshold: Not used for regression (kept for API compatibility)
        
    Returns:
        Dictionary containing predictions and evaluation metrics
    """
    results = []
    
    for probe in probes:
        predictions = probe.predict(test_features)
        
        # Calculate regression metrics
        r2_score = probe.score(test_features, test_labels)
        mse = np.mean((test_labels - predictions)**2)
        mae = np.mean(np.abs(test_labels - predictions))
        
        results.append({
            "r2_score": r2_score,
            "mse": mse,
            "mae": mae
        })
        
    return results

def compute_metrics(predictions, true_labels):
    """
    Compute regression metrics
    """
    mse = np.mean((true_labels - predictions)**2)
    mae = np.mean(np.abs(true_labels - predictions))
    r2 = 1 - (np.sum((true_labels - predictions)**2) / np.sum((true_labels - np.mean(true_labels))**2))
    return mse, mae, r2

    
    
# Create training results table
def display_training_results(results):
    table = PrettyTable()
    table.title = "Training Results"

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("Active Features", [result.get("active_features", 0) for result in results])
    table.add_column("Sparsity", [result.get("sparsity", 0) for result in results])
    table.add_column("R2 Score", [result.get("r2_score", 0) for result in results])
    table.add_column("MSE", [result.get("mse", 0) for result in results])
    return table
    

# Create testing results table
def display_testing_results(results):
    table = PrettyTable()
    table.title = "Testing Results"
    # Add rows with your testing metrics one column for each result

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("R2 Score", [result.get("r2_score", 0) for result in results])
    table.add_column("MSE", [result.get("mse", 0) for result in results])
    table.add_column("MAE", [result.get("mae", 0) for result in results])



    
    return table
    
    
    
    

def save_results_to_file(train_table, test_table, filename="results.txt"):
    """
    Save the training and testing results tables to a text file
    
    Args:
        train_table: PrettyTable object containing training results
        test_table: PrettyTable object containing testing results
        filename: Name of the file to save results to
    """
    output_path = f"/home/woody/b114cb/b114cb23/boxo/sae_oracle/{filename}"
    
    with open(output_path, "w") as f:
        f.write(str(train_table) + "\n\n")
        f.write(str(test_table) + "\n")
    
    print(f"Results saved to {output_path}")

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
        model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
        sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_25/sae_training_iter_0/final"

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
