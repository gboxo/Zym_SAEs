import torch
import pickle as pkl
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os
import pandas as pd
from argparse import ArgumentParser


args = ArgumentParser()
args.add_argument("--cfg_path", type=str)
args = args.parse_args()

path = args.cfg_path


output_dir = path["output_dir"]
seqs_path = path["seqs_path"]




fasta_files = [os.path.join(seqs_path, elem) for elem in os.listdir(seqs_path) if elem.endswith(".fasta")]
all_dfs = []
for fasta_file in fasta_files:
    with open(fasta_file, "r") as f:
        data = f.read()
    data = data.split("\n")
    data = [dat for dat in data if dat != ""]
    data = [elem.split(",")[1] for elem in data]
    data = [seq.strip("3. 2. 1. 1 <sep> <start>").strip("<end>") for seq in data]
    sequences = [seq.replace(" ", "") for seq in data]
    ids = list(range(len(sequences)))
    ids_seqs = list(zip(ids, sequences))
    df = pd.DataFrame(ids_seqs, columns=["id", "sequence"])
    df["feature"] = fasta_file.split("/")[-1].split(".")[0].split("_")[-1]
    all_dfs.append(df)

all_dfs = pd.concat(all_dfs)




##### Load the module ESM ######

print("Loading tokenizer and model")
tokenizer_esm = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold") # Download tokenizer
model_esm = EsmForProteinFolding.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold")  # Download model
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print("Moving the device to ", device_name)
device = torch.device(device_name)
model_esm = model_esm.to(device)
model_esm.eval()


for _, row in all_dfs.iterrows():
    id = row['id']
    seq = row['sequence']
    feature = row['feature']
    output_dir = f""
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{feature}_{id}.pdb"

    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"Skipping {id} - PDB file already exists")
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
