import torch
import os
import matplotlib.pyplot as plt
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from Bio.PDB import PDBParser, DSSP

"""
Questions that I want to answer:
    - For each feature in how many sequences it appears at least once
    - For each feature how much it's firing is correlated with the position
    - For each feature when it appears once how many times on average does it fire
    - For each feature that fires more thatn once in a sequence is the firing pattern sequential?

"""


# Load the MODEL and SAE
tokenizer, model = load_model("AI4PD/ZymCTRL")
model = get_ht_model(model, model.config).to("cuda")
sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg, sae = load_sae(sae_path)
thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
sae.to("cuda")
jump_relu = convert_to_jumprelu(sae, thresholds)
jump_relu.eval()



# Load the eval set
pdb = "/users/nferruz/gboxo/output_iteration3/PDB/"

seq_gen = "/users/nferruz/gboxo/seq_gen_3.2.1.1_iteration3.fasta"

with open(seq_gen, "r") as f:
    test_set = f.read()
    test_set = test_set.split(">")
    test_set = [elem for elem in test_set if len(elem) > 0]
    ids = [elem.split("\n")[0].split("\t")[0] for elem in test_set]
    seqs = [elem.split("\n")[1] for elem in test_set]
    all = list(zip(ids, seqs))
    df = pd.DataFrame(all, columns=["id", "seq"])




folded_seq_ids = os.listdir(pdb)
lf = lambda x: x.replace(".pdb","")
folded_seq_ids = list(map(lf, folded_seq_ids))


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

    for residue in dssp:
        chain_id = residue[0]
        res_id = residue[1]
        ss_type = residue[2]  # Secondary structure code

        if ss_type == "H":
            alpha_helix_residues.append((chain_id, res_id))
        elif ss_type == "E":
            beta_sheet_residues.append((chain_id, res_id))


    return alpha_helix_residues, beta_sheet_residues

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
# Example usage:
id = "3.2.1.1_0_0_iteration3"
pdb_path = os.path.join(pdb, id+".pdb")
alpha_helix_residues, beta_sheet_residues = get_secondary_structure_residues(pdb_path)


# Get DISTINCT secondary structures
alpha_helix_distinct = get_distinct_secondary_structures(alpha_helix_residues)
beta_sheet_distinct = get_distinct_secondary_structures(beta_sheet_residues)

print(alpha_helix_distinct)
print(beta_sheet_distinct)
