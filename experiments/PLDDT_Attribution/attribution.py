
# %%
import os
os.environ["PYTHONPATH"] = "/users/nferruz/gboxo/crg_boxo/"

# %%
import pickle as pkl
import pandas as pd
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LassoCV
import torch
import numpy as np
from scipy.sparse import coo_matrix, vstack
from prettytable import PrettyTable
from tqdm import tqdm
from src.utils import get_paths
from matplotlib import pyplot as plt
# %%






def get_sequences():

    df = pd.read_csv("Data/PLDDT_Attribution_Data/seqs_plddt.csv")
    seqs = df["sequence"].values
    plddts = df["plddt"].values
    with open("Data/PLDDT_Attribution_Data/sequences.txt", "w") as f:
        for seq in seqs:
            f.write(seq + "\n")



    return seqs, plddts




def get_activations_and_log_probs( model, tokenizer, sequence):
    sequence = "1.3.3.18" + "<sep>" +  "<start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("26.hook_resid_pre")
        logits, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.26.hook_resid_pre"]
        log_probs = compute_log_probbilities(inputs, logits)
    return log_probs, activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]
    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def compute_log_probbilities(tokens, logits):
    all_log_probs = []
    for i, token in enumerate(tokens):
        log_probs = torch.log_softmax(logits[0, i] + 1e-10, dim=-1)
        log_prob = log_probs[token].cpu().numpy()
        all_log_probs.append(log_prob)
    return all_log_probs



def get_adjusted_rewards(all_log_probs, all_plddts):
    all_adjusted_rewards = []
    for log_probs, plddt in zip(all_log_probs, all_plddts):
        all_adjusted_rewards.append(np.multiply(log_probs[0], plddt))
    return all_adjusted_rewards

def get_all_features(model, sae, tokenizer, sequences):
    all_features = []
    all_log_probs = []
    for sequence in tqdm(sequences):
        log_probs, activations = get_activations_and_log_probs(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features)
        all_log_probs.append(log_probs)
        del activations, features
        torch.cuda.empty_cache()
    return all_log_probs, all_features

def obtain_features(text_path):
    """
    Obtain features from natural sequences
    """
    assert text_path.endswith(".txt"), "Text file must end with .txt"
    file_name = text_path.split("/")[-1].split(".")[0].split("_")[0]
    assert len(file_name) > 0, "File name is empty"


    with open(text_path, "r") as f:
        sequences = f.read()
        sequences = sequences.split("\n")

    log_probs, features = get_all_features(model,jump_relu, tokenizer, sequences)
    log_probs = get_adjusted_rewards(log_probs, plddts)



    random_indices = np.random.permutation(len(features)-1)
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    test_indices = random_indices[int(len(random_indices)*0.8):]

    train_features = [features[i] for i in train_indices]
    train_features = vstack(train_features)
    test_features = [features[i] for i in test_indices]
    train_log_probs = [log_probs[i] for i in train_indices]
    train_log_probs = np.concatenate(train_log_probs)
    test_log_probs = [log_probs[i] for i in test_indices]



    os.makedirs(f"Data/PLDDT_Attribution_Data/features", exist_ok=True)
    np.savez(f"Data/PLDDT_Attribution_Data/features/{file_name}_features_train.npz",train_features)
    np.save(f"Data/PLDDT_Attribution_Data/features/{file_name}_features_test.npy",test_features)

    np.save(f"Data/PLDDT_Attribution_Data/features/{file_name}_log_probs_train.npy",train_log_probs)
    pkl.dump(test_log_probs, open(f"Data/PLDDT_Attribution_Data/features/{file_name}_log_probs_test.pkl", "wb"))



    del features, train_features, test_features, random_indices, train_indices, test_indices, log_probs
    torch.cuda.empty_cache()

def load_features(train_path, test_path, train_log_probs_path, test_log_probs_path):
    """
    Load features from a file
    """
    assert train_path.endswith(".npz") or train_path.endswith(".npy"), "File must end with .npz or .npy"
    assert test_path.endswith(".npz") or test_path.endswith(".npy"), "File must end with .npz or .npy"
    assert train_log_probs_path.endswith(".npy"), "File must end with .npy"
    assert test_log_probs_path.endswith(".pkl"), "File must end with .pkl"
    file_name = train_path.split("/")[-1].split(".")[0]
    assert len(file_name) > 0, "File name is empty"

    train_features = np.load(train_path, allow_pickle=True)
    test_features = np.load(test_path, allow_pickle=True)
    train_log_probs = np.load(train_log_probs_path, allow_pickle=True)
    with open(test_log_probs_path, "rb") as f:
        test_log_probs = pkl.load(f)

    train_features = train_features["arr_0"].tolist()
    test_features = test_features.tolist()

    return train_features, test_features, train_log_probs, test_log_probs


def train_logistic_probe(train_features, train_log_probs, test_features, test_log_probs):
    # Concatennate  COO
    X_train = train_features
    X_test = vstack(test_features)
    y_train = train_log_probs 
    y_test = np.concatenate(test_log_probs, axis = 0) 
    nan_ind_train = np.where(np.isnan(y_train))[0]
    nan_ind_test = np.where(np.isnan(y_test))[0]

    y_train[nan_ind_train] = 0
    y_test[nan_ind_test] = 0



    y_train = y_train < -1100
    y_test = y_test < -1100


    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-4, -2.5, 10)):
        lr_model = LogisticRegressionCV(penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1, cv=5)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[0]
        probes.append(lr_model)
        y_pred = lr_model.predict(X_test)
        probs = lr_model.predict_proba(X_test)
        metrics = compute_metrics_logistic(probs[:, 1], y_pred, y_test)
        results.append({
            "active_features": len(active_features),
            "sparsity": sparsity,
            **metrics
        })
    return probes, results

def compute_metrics_logistic(probabilities, predictions, true_labels):
    accuracy = accuracy_score(predictions, true_labels)
    precision = precision_score(predictions, true_labels)
    recall = recall_score(predictions, true_labels)
    f1 = f1_score(predictions, true_labels)
    roc_auc = roc_auc_score(true_labels, probabilities)
    d = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    return d




def train_linear_probe(train_features, train_log_probs, test_features, test_log_probs):
    # Concatennate  COO
    X_train = train_features
    X_test = vstack(test_features)
    y_train = train_log_probs 
    y_test = np.concatenate(test_log_probs, axis = 0)

    nan_ind_train = np.where(np.isnan(y_train))[0]
    nan_ind_test = np.where(np.isnan(y_test))[0]

    y_train[nan_ind_train] = 0
    y_test[nan_ind_test] = 0
  
    

    results = []
    probes =[]
    for sparsity in tqdm([0.01,0.015, 0.02, 0.025, 0.03]):
        lr_model = LassoCV(alphas=[1/sparsity], n_jobs=-1, cv=5)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[0]
        probes.append(lr_model)
        y_pred = lr_model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        results.append({
            "active_features": len(active_features),
            "sparsity": sparsity,
            **metrics
        })
    best_result = max(results, key=lambda x: x["explained_variance"])
    return probes, results


def compute_metrics(predicted_scores, true_scores):
    """
    Compute metrics for Linear Regression
    
    Args:
        predicted_scores: Model predictions
        true_scores: Ground truth values
        
    Returns:
        Dictionary containing:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error) 
        - MAE (Mean Absolute Error)
        - R2 Score (Coefficient of determination)
        - Explained Variance Score
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
    import numpy as np
    
    mse = mean_squared_error(true_scores, predicted_scores)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_scores, predicted_scores)
    r2 = r2_score(true_scores, predicted_scores)
    ev_score = explained_variance_score(true_scores, predicted_scores)
    
    return {
        "mse": mse,
        "rmse": rmse, 
        "mae": mae,
        "r2": r2,
        "explained_variance": ev_score
    }





# Create training results table
def display_training_results(results):
    table = PrettyTable()
    table.title = "Training Results"

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("Active Features", [result.get("active_features", 0) for result in results])
    table.add_column("Sparsity", [result.get("sparsity", 0) for result in results])
    table.add_column("MSE", [result.get("mse", 0) for result in results])
    table.add_column("RMSE", [result.get("rmse", 0) for result in results])
    table.add_column("MAE", [result.get("mae", 0) for result in results])
    table.add_column("R2", [result.get("r2", 0) for result in results])
    table.add_column("Explained Variance", [result.get("explained_variance", 0) for result in results])
    return table

# Create training results table
def display_training_results_logistic(results):
    table = PrettyTable()
    table.title = "Training Results"

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("Active Features", [result.get("active_features", 0) for result in results])
    table.add_column("Sparsity", [result.get("sparsity", 0) for result in results])
    table.add_column("Accuracy", [result.get("accuracy", 0) for result in results])
    table.add_column("ROC AUC", [result.get("roc_auc", 0) for result in results])
    return table

    # %%

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
        sequences, plddts = get_sequences()
        obtain_features("Data/PLDDT_Attribution_Data/sequences.txt")



    if True:

        sequences, plddts = get_sequences()
        train_features, test_features, train_adjusted_rewards, test_adjusted_rewards = load_features("Data/PLDDT_Attribution_Data/features/sequences_features_train.npz", "Data/PLDDT_Attribution_Data/features/sequences_features_test.npy", "Data/PLDDT_Attribution_Data/features/sequences_log_probs_train.npy", "Data/PLDDT_Attribution_Data/features/sequences_log_probs_test.pkl")



        # ======= Train Linear Probes =======

        #probes, train_results = train_linear_probe(train_features, train_adjusted_rewards, test_features, test_adjusted_rewards)
        probes, train_results = train_logistic_probe(train_features, train_adjusted_rewards, test_features, test_adjusted_rewards)
        # Display all three tables
        #training_table = display_training_results(train_results).get_string()
        training_table_logistic = display_training_results_logistic(train_results).get_string()
        os.makedirs("Data/PLDDT_Attribution_Data/results", exist_ok=True)

        with open("Data/PLDDT_Attribution_Data/results/training_table_logistic.txt", "w") as f:
            f.write(training_table_logistic)
