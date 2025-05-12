import pandas as pd
import pickle as pkl
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import argparse
from src.tools.oracles.oracles_utils import load_config
from torch.utils.data import DataLoader
from datasets import Dataset
import numpy as np
from sklearn.linear_model import LinearRegression




# Load the Linear Regression model

with open("/home/woody/b114cb/b114cb23/boxo/esm_lr/mlp_regressor.pkl", "rb") as f:
    lr = pkl.load(f)





def read_sequence_from_file(path):
    """
    Reads **all** sequences in a .txt (multiple FASTA‐style entries).
    Each header line is of the form:
      >ID,… <sep> <start> A B C … <end>

    Returns:
      A dictionary mapping sequence IDs to sequence strings (with spaces removed)
    """
    seqs = {}
    curr = ""
    curr_id = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # If we were collecting a sequence, save it before starting a new one
                if curr and curr_id:
                    seqs[curr_id] = curr
                    curr = ""
                # Get the ID from the header line
                curr_id = line[1:].split(",")[0]
                # If this header has the entire seq on the same line:
                if "<start>" in line and "<end>" in line:
                    block = line.split("<start>", 1)[1].split("<end>", 1)[0]
                    curr = block.replace(" ", "")
                continue

            # Continue accumulating sequence tokens
            curr += line.replace(" ", "")

    # At EOF, flush last sequence
    if curr and curr_id:
        seqs[curr_id] = curr

    return seqs



class SequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = torch.cat([seq["input_ids"] for seq in tokenized_sequences])
        self.attention_mask = torch.cat([seq["attention_mask"] for seq in tokenized_sequences])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }



def main():

    parser = argparse.ArgumentParser(
        description="Predict activities on a directory of ablated sequences"
    )
    parser.add_argument(
        "--cfg_path", type=str, required=True,
        help="Path to your YAML/JSON config (for oracle paths)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for DataLoader"
    )
    args = parser.parse_args()

    # Load config (to get oracle checkpoint paths)
    config = load_config(args.cfg_path)

    # ========== 1) Read all sequences (flatten multiple entries per file) ==========
    records = []
    if False:

        for fn in sorted(os.listdir(config["paths"]["seqs_path"])):
            if not fn.endswith(".txt"):
                continue
            full = os.path.join(config["paths"]["seqs_path"], fn)

            seqs = read_sequence_from_file(full)
            keys = list(seqs.keys())
            seqs = list(seqs.values())
            base = os.path.splitext(fn)[0]

            # each <start>…<end> becomes its own record
            for idx_seq, seq_str in enumerate(seqs):
                records.append({
                    "name":      base,
                    "index":     keys[idx_seq],   # per‐file sequence index
                    "sequence":  seq_str
                })
    else:
        full = config["paths"]["seqs_path"]

        seqs = read_sequence_from_file(full)
        print(seqs)
        keys = list(seqs.keys())
        seqs = list(seqs.values())
        fn = full.split("/")[-1]
        base = os.path.splitext(fn)[0]

        # each <start>…<end> becomes its own record
        for idx_seq, seq_str in enumerate(seqs):
            records.append({
                "name":      base,
                "index":     keys[idx_seq],   # per‐file sequence index
                "sequence":  seq_str
            })

    # ========== 2) Tokenize & build DataLoader ==========
    # We only need one tokenizer, so load first oracle
    oracle1_ckpt = config["paths"]["oracle_path1"]
    oracle1_weights = config["paths"]["checkpoint_path1"]

    # ========== 2) Tokenize & build DataLoader ==========
    # We only need one tokenizer, so load first oracle
    tokenizer = AutoTokenizer.from_pretrained(oracle1_ckpt)
    model = AutoModel.from_pretrained(
        oracle1_ckpt,
        torch_dtype=torch.float16
    )

    # Tokenize all sequences in a batch
    tokenized = tokenizer(
        [r["sequence"] for r in records],
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


    tensor_dataset = SequenceDataset([{
        "input_ids": tokenized["input_ids"][i : i+1],
        "attention_mask": tokenized["attention_mask"][i : i+1]
    } for i in range(len(tokenized["input_ids"]))])
    loader = DataLoader(tensor_dataset, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device).eval()
    preds1 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle1"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            # Get the hidden states from layer 33 (last layer)
            for i in range(len(batch["input_ids"])):
                layer_33_output = outputs.hidden_states[-1][i]
                mask = attention_mask[i]
                layer_33_output = layer_33_output[mask == 1]
                preds1.append(layer_33_output.mean(dim=0))
    # free GPU
    torch.cuda.empty_cache()
    # ========== 3) Predict activities ==========

    preds1 = torch.stack(preds1)
    X = preds1.cpu().numpy()

    y_pred = lr.predict(X)

    # ========== 4) Build DataFrame & Save ==========
    df = pd.DataFrame({
        "name":        [r["name"]       for r in records],
        "index":       [r["index"]      for r in records],
        "prediction": y_pred
    })

    output_path = config["paths"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")

if __name__ == "__main__":
    main()








