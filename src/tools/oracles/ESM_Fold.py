import torch
import pickle as pkl
from transformers import AutoTokenizer, EsmForProteinFolding
import argparse
import os
import pandas as pd
from oracles_utils import load_config
from argparse import ArgumentParser






def main(fasta_path, output_path):

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
        output_file = f"{output_path}/{name}.pdb"

        # Skip if file already exists
        if os.path.exists(output_file):
            print(f"Skipping {name} - PDB file already exists")
            continue

        print(f"Processing sequence {id}")
        print(f"Sequence: {seq}")
        with torch.no_grad():
            output = model_esm.infer_pdb(seq)
            torch.cuda.empty_cache()
            with open(output_file, "w") as f:
                f.write(output)
            del output
            torch.cuda.empty_cache()


if __name__ == "__main__":
    arguments = ArgumentParser()
    arguments.add_argument("--cfg_path", type=str)
    arguments.add_argument("--iteration_num", type=int)
    args = arguments.parse_args()
    cfg_path = args.cfg_path
    iteration_num = args.iteration_num
    label = args.label

    config = load_config(cfg_path)
    fasta_path = config["paths"]["fasta_path"].format(label, iteration_num)
    output_path = config["paths"]["output_path"].format(iteration_num)






    ##### Load the module ESM ######
    print("Loading tokenizer and model")
    tokenizer_esm = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold") # Download tokenizer
    model_esm = EsmForProteinFolding.from_pretrained("/home/woody/b114cb/b114cb23/models/esm_fold")  # Download model
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print("Moving the device to ", device_name)
    device = torch.device(device_name)
    model_esm = model_esm.to(device)
    model_esm.eval()



    main(fasta_path, output_path)
