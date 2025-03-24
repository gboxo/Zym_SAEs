import torch
import pickle as pkl
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os
import pandas as pd


##### Load the module ESM ######
print("Loading tokenizer and model")
tokenizer_esm = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Moving the device to ", device_name)
device = torch.device(device_name)
model_esm = model_esm.to(device)
model_esm.eval()
parser = argparse.ArgumentParser()
parser.add_argument("--iteration_num", type=int)
parser.add_argument("--label", type=str)
parser.add_argument("--procedure", type=str)
args = parser.parse_args()
iteration_num = args.iteration_num
ec_label = args.label
ec_label = ec_label.strip()
data_iteration = iteration_num

procedure = args.procedure #["diffing", "steering", "ablation"]

if procedure == "diffing": # For diffing dataset
    fasta_path = f"/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_{ec_label}_iteration{iteration_num}.fasta"

elif procedure == "steering": # For steering dataset
    # Load top correlations to get feature indices
    path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/correlations/top_correlations_M{iteration_num}_D{data_iteration}.pkl"
    with open(path, "rb") as f:
        top_correlations = pkl.load(f)
    feature_indices = top_correlations["feature_indices"]

    all_fasta_paths = []        
    for steering_feature in feature_indices:
        fasta_path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/steering/M{iteration_num}_D{data_iteration}/steering_feature_{steering_feature}.txt"
        all_fasta_paths.append(fasta_path)

elif procedure == "ablation": # For ablation dataset
    # Load top correlations to get feature indices  
    path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/correlations/top_correlations_M{iteration_num}_D{data_iteration}.pkl"
    with open(path, "rb") as f:
        top_correlations = pkl.load(f)
    feature_indices = top_correlations["feature_indices"]

    all_fasta_paths = []
    for ablation_feature in feature_indices:
        fasta_path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/ablation/M{iteration_num}_D{data_iteration}/ablation_feature_{ablation_feature}.txt"
        all_fasta_paths.append(fasta_path)
    
    



# Put sequences into dictionary
if procedure in ["steering", "ablation"]:
    all_dfs = []
    for fasta_path, feature in zip(all_fasta_paths, feature_indices):
        with open(fasta_path, "r") as f:
            data = f.read()
        data = data.split("\n")
        data = [dat for dat in data if dat != ""]
        ids = [dat for dat in data if dat.startswith(">")]
        sequences = [dat for dat in data if dat.startswith(">") == False]
        sequences = [seq.strip("3. 2. 1. 1 <sep> <start>").strip("<end>") for seq in sequences]
        ids_seqs = list(zip(ids, sequences))
        df_i = pd.DataFrame(ids_seqs, columns=["id", "sequence"])
        df_i["feature"] = feature
        all_dfs.append(df_i)

    df = pd.concat(all_dfs)

    for _, row in df.iterrows():
        id = row['id']
        seq = row['sequence'].replace(" ", "")
        feat = row['feature']
        name = id[1:]
        name = name.split("\t")[0]
        name = name + "_" + procedure + "_" + str(feat)
        output_dir = f"/home/woody/b114cb/b114cb23/boxo/outputs_{procedure}/output_iterations{iteration_num}/PDB"
        output_file = f"{output_dir}/{name}.pdb"

        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Skipping {name} - PDB file already exists")
            continue

        print(f"Processing sequence {id}")
        print(f"Feature: {feat}")
        print(f"Sequence: {seq}")
        if len(seq) > 600:
            print(f"Skipping {name} - Sequence too long")
            continue

        with torch.no_grad():
            output = model_esm.infer_pdb(seq)
            torch.cuda.empty_cache()
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(output)
            del output
            torch.cuda.empty_cache()
    del model_esm

else:
    with open(fasta_path, "r") as f:
        data = f.read()
    data = data.split("\n")
    data = [dat for dat in data if dat != ""]
    ids = [dat for dat in data if dat.startswith(">")]
    sequences = [dat for dat in data if dat.startswith(">") == False]
    sequences = [seq.strip("3. 2. 1. 1 <sep> <start>").strip("<end>") for seq in sequences]
    ids_seqs = list(zip(ids, sequences))
    df = pd.DataFrame(ids_seqs, columns=["id", "sequence"])

    for _, row in df.iterrows():
        id = row['id']
        seq = row['sequence'].replace(" ", "")
        name = id[1:]
        name = name.split("\t")[0]
        output_dir = f"/home/woody/b114cb/b114cb23/boxo/outputs/output_iterations{iteration_num}/PDB"
        output_file = f"{output_dir}/{name}.pdb"

        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Skipping {name} - PDB file already exists")
            continue

        print(f"Processing sequence {id}")
        print(f"Sequence: {seq}")
        with torch.no_grad():
            output = model_esm.infer_pdb(seq)
            torch.cuda.empty_cache()
            os.makedirs(output_dir, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(output)
            del output
            torch.cuda.empty_cache()
