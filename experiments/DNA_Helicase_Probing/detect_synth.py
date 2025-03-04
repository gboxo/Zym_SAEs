# %%
import json
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
from src.utils import get_paths
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

# %%
def get_natural_and_synth_sequences(brenda_path):


    with open(brenda_path, "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    ec_numbers = [elem.split("<sep>")[0] for elem in test_set if len(elem.split("<sep>")) > 1]
    sequences = [elem.split("<sep>")[1] for elem in test_set if len(elem.split("<sep>")) > 1]
    ec_numbers = [elem for elem in ec_numbers if len(elem) > 0]
    indices = [i for i,elem in enumerate(ec_numbers) if elem == "3.6.4.12"]
    natural_sequences = [sequences[i].strip("<start>").strip("<end>").strip("<|endoftext|").strip("<end>") for i in indices]


    files = os.listdir("Data/DNA_Helicase_Data/DNA_Helicase_generation/")
    files = [file for file in files if file.endswith(".fasta")]
    synth_sequences = []
    for file in files:
        with open(f"Data/DNA_Helicase_Data/DNA_Helicase_generation/{file}", "r") as f:
            seq = f.read()
            seq = seq.split("\n")[1].strip("<en")
        synth_sequences.append(seq)

    with open("Data/DNA_Helicase_Data/natural_sequences.txt", "w") as f:
        for seq in natural_sequences:
            f.write(seq + "\n")

    with open("Data/DNA_Helicase_Data/synth_sequences.txt", "w") as f:
        for seq in synth_sequences:
            f.write(seq + "\n")



    return natural_sequences, synth_sequences


def get_activations( model, tokenizer, sequence):
    sequence = "3.6.4.12<sep>" + sequence
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
    for sequence in sequences:
        activations = get_activations(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features)
        del activations, features
        torch.cuda.empty_cache()
    return all_features

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

    features = get_all_features(model,jump_relu, tokenizer, sequences)
    random_indices = np.random.permutation(len(features))
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    test_indices = random_indices[int(len(random_indices)*0.8):]

    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]

    train_features = vstack(train_features)
    os.makedirs(f"Data/DNA_Helicase_Data/features", exist_ok=True)
    np.savez(f"Data/DNA_Helicase_Data/features/{file_name}_features_train.npz",train_features)
    np.save(f"Data/DNA_Helicase_Data/features/{file_name}_features_test.npy",test_features)
    del features, train_features, test_features, random_indices, train_indices, test_indices
    torch.cuda.empty_cache()

def load_features(train_path, test_path):
    """
    Load features from a file
    """
    assert train_path.endswith(".npz") or train_path.endswith(".npy"), "File must end with .npz or .npy"
    assert test_path.endswith(".npz") or test_path.endswith(".npy"), "File must end with .npz or .npy"
    file_name = train_path.split("/")[-1].split(".")[0]
    assert len(file_name) > 0, "File name is empty"

    train_natural_features = np.load(train_path, allow_pickle=True)
    test_natural_features = np.load(test_path, allow_pickle=True)

    train_natural_features = train_natural_features["arr_0"].tolist()
    test_natural_features = test_natural_features.tolist()
    return train_natural_features, test_natural_features

def train_linear_probe(train_natural_features, train_synth_features, test_natural_features, test_synth_features):
    # Concatennate  COO

    X_train = vstack((train_natural_features, train_synth_features))
    X_test = vstack((vstack(test_natural_features), vstack(test_synth_features)))

    y_train = np.concatenate((np.zeros(train_natural_features.shape[0]), np.ones(train_synth_features.shape[0])), axis=0)
    y_test = np.concatenate((np.zeros(vstack(test_natural_features).shape[0]), np.ones(vstack(test_synth_features).shape[0])), axis=0)
    

    results = []
    probes =[]
    for sparsity in tqdm([0.00001,0.0001,0.001,0.01]):
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
    label_dict = {"natural": 0, "synth": 1}
    
    for probe in probes:
        all_predictions = []
        all_probs = []
        true_labels = []
        for label, test_features in [("natural", test_natural_features), ("synth", test_synth_features)]:
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





# Create training results table
def display_training_results(results):
    table = PrettyTable()
    table.title = "Training Results"

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("Active Features", [result.get("active_features", 0) for result in results])
    table.add_column("Sparsity", [result.get("sparsity", 0) for result in results])
    table.add_column("Accuracy", [result.get("accuracy", 0) for result in results])
    table.add_column("ROC AUC", [result.get("roc_auc", 0) for result in results])
    return table
    

# Create testing results table
def display_testing_results(results):
    table = PrettyTable()
    table.title = "Testing Results"
    # Add rows with your testing metrics one column for each result

    table.add_column("Model", [i for i in range(len(results))])
    table.add_column("Accuracy", [result.get("accuracy", 0) for result in results])
    table.add_column("Precision", [result.get("precision", 0) for result in results])
    table.add_column("Recall", [result.get("recall", 0) for result in results])
    table.add_column("F1 Score", [result.get("f1", 0) for result in results])
    table.add_column("ROC AUC", [result.get("roc_auc", 0) for result in results])



    
    return table


    # %%

if __name__ == "__main__":
    paths = get_paths()
    brenda_path = paths.mini_brenda
    natural_sequences, synth_sequences = get_natural_and_synth_sequences(brenda_path)
    model_path = paths.model_path
    if True:
        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        sae_path = paths.sae_path
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        torch.cuda.empty_cache()


    # ==== NATURAL =======
    #obtain_features("../Data/DNA_Helicase_Data/natural_sequences.txt")

    # ==== SYNTHETIC =======
    #obtain_features("../Data/DNA_Helicase_Data/synth_sequences.txt")
    train_natural_features, test_natural_features = load_features("Data/DNA_Helicase_Data/features/natural_features_train.npz", "Data/DNA_Helicase_Data/features/natural_features_test.npy")
    train_synth_features, test_synth_features = load_features("Data/DNA_Helicase_Data/features/synth_features_train.npz", "Data/DNA_Helicase_Data/features/synth_features_test.npy")

    # ======= Train Linear Probes =======

    probes, train_results = train_linear_probe(train_natural_features, train_synth_features, test_natural_features, test_synth_features)
    test_results = test_linear_probe(probes, test_natural_features, test_synth_features)

    # Display all three tables
    training_table = display_training_results(train_results).get_string()
    testing_table = display_testing_results(test_results).get_string()

    os.makedirs("results", exist_ok=True)

    with open("Data/DNA_Helicase_Data/results/training_table.txt", "w") as f:
        f.write(training_table)
    with open("Data/DNA_Helicase_Data/results/testing_table.txt", "w") as f:
        f.write(testing_table)
