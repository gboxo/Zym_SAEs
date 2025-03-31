"""
This script contains the logic to do secondary structure annotation with SAEs

COMPONENTS:
    - Feature extraction with ZymCTRL and SAEs
    - Secondary structure annotation with DSSP
    - The featuers are aggregated across contiguous secondary structures
    - Two methods for chracterizing features:
        - Max mean difference 
            - We take the mean of features for a given structure adn subtract the mean of features for the other structure
            - We take the maximum of the absolute values of the differences
        - Sparse Probing
"""



import torch
import os
import numpy as np
from experiments.annotation.utils import get_all_features, load_features, get_secondary_structure_residues, get_distinct_secondary_structures
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from tqdm import tqdm
import pickle as pkl
from concurrent.futures import ThreadPoolExecutor
import warnings
from src.utils import get_paths
from Bio.PDB import PDBParser
from scipy.sparse import vstack

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


def get_tokenized_seqs(info):
    """
    Extract features from a list of amino acid sequences.
    
    Args:
        Dictionary with information about the sequences, EC LABELS, and secondary structures
    
    Returns:
        list: List of features for each amino acid sequence
    """
    tokenized_features = {} 
    for key,val in info.items():
        ec = val["ec"][0]
        if ec == None:
            continue
        seq = ec + "<sep><start>" + val["aa_seq"] + "<end>"
        tokenized_features[key] = tokenizer.encode(seq, return_tensors="pt", truncation=True, max_length = 512).to("cuda")
    return tokenized_features


def get_all_structures(pdb_path):

    dirs = os.listdir(path)
    all_paths = [path + elem for elem in dirs]
    all_ecs = {}
    for file in all_paths:
        name = file.split("/")[-1].split(".")[0]
        structure = PDBParser().get_structure("enzyme", file)
        header = structure.header
        compound = header["compound"]
        x = []
        for key, val in compound.items():
            ec = val.get("ec", None)
            x.append(ec)
        all_ecs[name] = x 

    with open("Data/annotation/ec.pkl", "wb") as f:
        pkl.dump(all_ecs, f)
    # Load existing results
    processed_results = load_structures()
    processed_names = {info['name'] for info in processed_results if info is not None}
    
    # Filter out already processed paths
    remaining_paths = [p for p in all_paths if p.split("/")[-1].split(".")[0] not in processed_names]
    
    if remaining_paths:
        print(f"Processing {len(remaining_paths)} remaining structures...")
        with ThreadPoolExecutor(max_workers=20) as executor:
            for batch in tqdm([remaining_paths[i:i+100] for i in range(0, len(remaining_paths), 100)]):
                new_results = list(executor.map(get_structures_and_seq, batch))
                processed_results.extend(new_results)
                # Save intermediate results after each batch
                save_structures(processed_results)
        
        print("Processing complete!")
    else:
        print("All structures have already been processed!")

def get_structures_and_seq(pdb_path):
    try:
        name = pdb_path.split("/")[-1].split(".")[0]
        alpha_helix_residues, beta_sheet_residues, aa_seq = get_secondary_structure_residues(pdb_path)
        alpha_helix_residues = get_distinct_secondary_structures(alpha_helix_residues)
        beta_sheet_residues = get_distinct_secondary_structures(beta_sheet_residues)
        d_info = {
                "name": name,
                "alpha_helix_residues": alpha_helix_residues,
                "beta_sheet_residues": beta_sheet_residues,
                "aa_seq": aa_seq

                }
    except Exception as e:
        print(f"Error in {pdb_path}: {e}")
        return None
    return d_info    




def structure_labels(all_info):
    """
    Return a dictionary with the 1 hot encoded type of residue.
    We truncate up to 500 aa (structural propoerties for aa beyond this length are not taken into account)


    """
    all_positions = {}
    for key,val in all_info.items():
        alpha_helix_residues = val["alpha_helix_residues"]
        beta_sheet_residues = val["beta_sheet_residues"]
        alpha_positions = []
        for i in range(len(alpha_helix_residues)):
            alpha_positions.extend([elem[0] for elem in alpha_helix_residues[i] if elem[0] < 500])
        beta_positions = []
        for i in range(len(beta_sheet_residues)):
            beta_positions.extend([elem[0] for elem in beta_sheet_residues[i] if elem[0] < 500])

        positions = np.zeros(500)
        positions[alpha_positions] = 1
        positions[beta_positions] = -1 
        all_positions[key] = positions
    return all_positions






def load_structures():
    if not os.path.exists("Data/annotation/structures.pkl"):
        return []
    with open("Data/annotation/structures.pkl", "rb") as f:
        all_info = pkl.load(f)
    with open("Data/annotation/ec.pkl", "rb") as f:
        ecs = pkl.load(f)
    new_info = {}
    for info in all_info:
        if info:
            if info["name"] in ecs.keys():
                info["ec"] = ecs[info["name"]]
                new_info[info['name']] = info
            else:
                print(f"Error: {info['name']} not in ecs")
    return new_info

def save_structures(all_info):
    os.makedirs("Data/annotation", exist_ok=True)
    with open("Data/annotation/structures.pkl", "wb") as f:
        pkl.dump(all_info, f)

def obtain_features(model,jump_relu,tokenized_seqs,labels):
    """
    Obtain features from natural sequences
    """
    all_label = list(labels.values())

    features = get_all_features(model,jump_relu, tokenized_seqs)
    file_name = "secondary_structure_ec"
    os.makedirs(f"Data/annotation/features", exist_ok=True)
    with open(f"Data/annotation/features/{file_name}_features.pkl", "wb") as f:
        pkl.dump(features, f)
    with open(f"Data/annotation/features/{file_name}_labels.pkl", "wb") as f:
        pkl.dump(all_label, f)


    torch.cuda.empty_cache()


def classifier(train_features, test_features, train_labels, test_labels):
    """
    Train a classifier on the features
    """
    # Concatennate  COO

    X_train = train_features
    X_test = test_features
    y_train = train_labels
    y_test = test_labels

    X_test = vstack(X_test)








OBTAIN_STRUCTURES=False
OBTAIN_FEATURES=False


if __name__ == "__main__":
    if OBTAIN_STRUCTURES:
        path = "/users/nferruz/gboxo/pdb_files/"
        get_all_structures(path)
    else:
        all_info = load_structures()


    if OBTAIN_FEATURES: 
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

        tokenized_seqs = get_tokenized_seqs(all_info)
        labels = structure_labels(all_info)
        obtain_features(model,jump_relu,tokenized_seqs,labels)
    else:
        
        file_name = "secondary_structure_ec"
        with open(f"Data/annotation/features/{file_name}_features.pkl", "rb") as f:
            features = pkl.load(f)
        with open(f"Data/annotation/features/{file_name}_labels.pkl", "rb") as f:
            labels = pkl.load(f)
        alpha_features = []
        beta_features = []
        none_features = []
        for i in range(100):
            X = features[i].todense()[:500]
            l = len(X)
            y = labels[i][:l]
            y_alpha = y == 1
            y_beta = y == -1
            y_none = y == 0
            alpha_occurences = np.array(X[y_alpha]>0).sum(0)
            beta_occurences = np.array(X[y_beta]>0).sum(0)
            none_occurences = np.array(X[y_none]>0).sum(0)
            X_alpha = X[y_alpha].sum(axis=0)/alpha_occurences
            X_beta = X[y_beta].sum(axis=0)/beta_occurences
            X_none = X[y_none].sum(axis=0)/none_occurences
            X_alpha[np.isnan(X_alpha)] = 0
            X_beta[np.isnan(X_beta)] = 0
            X_none[np.isnan(X_none)] = 0
            alpha_features.append(X_alpha)
            beta_features.append(X_beta)
            none_features.append(X_none)
        alpha_features = np.array(alpha_features).squeeze().mean(axis=0)
        beta_features = np.array(beta_features).squeeze().mean(axis=0)
        none_features = np.array(none_features).squeeze().mean(axis=0)

        alpha_diff = np.abs(alpha_features - none_features)
        beta_diff = np.abs(beta_features - none_features)
        alpha_beta_diff = np.abs(alpha_features - beta_features)
        plt.plot(alpha_diff, label="Alpha Diff", alpha=0.5, color="blue")
        plt.plot(beta_diff, label="Beta", alpha=0.5, color="green")
        plt.legend()
        plt.show()


            




        





