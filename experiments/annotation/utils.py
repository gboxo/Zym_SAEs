from Bio.PDB import PDBParser, DSSP
import os
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from scipy.sparse import coo_matrix, vstack
from tqdm import tqdm
from Bio.SeqUtils import seq1  # Alternative method
# %%


def get_activations( model, inputs):
    with torch.no_grad():
        names_filter = lambda x: x.endswith("26.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.26.hook_resid_pre"]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]
    feature_acts = feature_acts.detach().cpu().numpy()[0,9:,:]
    sparse_feature_acts = coo_matrix(feature_acts)
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def get_all_features(model, sae, sequences):
    all_features = []
    for sequence in tqdm(sequences.values()):
        activations = get_activations(model, sequence)
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
        sequences = [seq for seq in sequences if seq != ""]

    features = get_all_features(model,jump_relu, sequences)
    random_indices = np.random.permutation(len(features))
    train_indices = random_indices[:int(len(random_indices)*0.8)]
    test_indices = random_indices[int(len(random_indices)*0.8):]

    train_features = [features[i] for i in train_indices]
    test_features = [features[i] for i in test_indices]

    train_features = vstack(train_features)
    os.makedirs(f"Data/Detect_Synth_Data/features", exist_ok=True)
    np.savez(f"Data/Detect_Synth_Data/features/{file_name}_features_train.npz",train_features)
    np.save(f"Data/Detect_Synth_Data/features/{file_name}_features_test.npy",test_features)
    del features, train_features, test_features, random_indices, train_indices, test_indices
    torch.cuda.empty_cache()


def load_features(train_path, test_path, train_labels_path, test_labels_path):
    """
    Load features from a file
    """
    assert train_path.endswith(".npz") or train_path.endswith(".npy"), "File must end with .npz or .npy"
    assert test_path.endswith(".npz") or test_path.endswith(".npy"), "File must end with .npz or .npy"
    file_name = train_path.split("/")[-1].split(".")[0]
    assert len(file_name) > 0, "File name is empty"

    train_features = np.load(train_path, allow_pickle=True)
    test_features = np.load(test_path, allow_pickle=True)

    train_labels = np.load(train_labels_path, allow_pickle=True)
    test_labels = np.load(test_labels_path, allow_pickle=True)

    train_features = train_features["arr_0"].tolist()
    test_features = test_features.tolist()
    return train_features, train_features, train_labels, test_labels


def get_secondary_structure_residues(pdb_path):
    """
    Extract residues belonging to alpha helices and beta sheets from a PDB file.
    
    Args:
        pdb_path (str): Path to the PDB file
    
    Returns:
        tuple: Two lists containing residues in alpha helices and beta sheets
    """
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_path)
    model = structure[0]  # Take the first model
    dssp = DSSP(model, pdb_path, dssp="mkdssp")

    # Extract residues in alpha helices (H) and beta sheets (E)
    alpha_helix_residues = []
    beta_sheet_residues = []
    seq = []

    for residue in dssp:
        chain_id = residue[0]
        res_id = residue[1]
        ss_type = residue[2]  # Secondary structure code
        seq.append(res_id)


        if ss_type == "H":
            alpha_helix_residues.append((chain_id, res_id))
        elif ss_type == "E":
            beta_sheet_residues.append((chain_id, res_id))
    seq = "".join(seq)


    return alpha_helix_residues, beta_sheet_residues, seq

def get_distinct_secondary_structures(structure_residues):

    temporal = []
    definitive = []

    for i in range(1,len(structure_residues)):
        if i == 1:
            temporal.append(structure_residues[i-1])


        resid_id = structure_residues[i][0]
        prev_resid_id = structure_residues[i-1][0]
        if resid_id - prev_resid_id == 1:
            temporal.append(structure_residues[i])
        else:
            definitive.append(temporal)
            temporal = []

    return definitive
