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



fasta_path = f"/home/woody/b114cb/b114cb23/boxo/kl_divergence/dataset.txt"




with open(fasta_path, "r") as f:
    data = f.read()
sequences = data.split("\n")
ids = [i for i in range(len(sequences))]
df = pd.DataFrame(zip(ids, sequences), columns=["id", "sequence"])
os.makedirs("/home/woody/b114cb/b114cb23/boxo/kl_divergence/PDB/", exist_ok=True)
for _, row in df.iterrows():
    id = row['id']
    seq = row['sequence'].replace(" ", "")
    output_dir = f"/home/woody/b114cb/b114cb23/boxo/kl_divergence/PDB/"
    output_file = f"{output_dir}/{id}.pdb"

    # Skip if file already exists


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
