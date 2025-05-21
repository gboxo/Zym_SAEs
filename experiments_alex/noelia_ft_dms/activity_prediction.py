import os
import numpy as np
import argparse
from tqdm import tqdm
import torch
import pandas as pd

from src.tools.oracles.oracles_utils import load_config
from src.tools.oracles.activity_prediction import load_model, SequenceDataset
from torch.utils.data import DataLoader



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
    if True:

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
    tokenizer, _ = load_model(
        oracle1_ckpt, oracle1_weights, num_labels=1
    )
    # Tokenize all sequences in a batch
    tokenized = tokenizer(
        [r["sequence"] for r in records],
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )


    print(tokenized["input_ids"])
    tensor_dataset = SequenceDataset([{
        "input_ids": tokenized["input_ids"][i : i+1],
        "attention_mask": tokenized["attention_mask"][i : i+1]
    } for i in range(len(records))])
    loader = DataLoader(tensor_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 3) Run Oracle #1 ==========
    _, model1 = load_model(
        oracle1_ckpt, oracle1_weights, num_labels=1
    )
    model1.to(device).eval()
    preds1 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle1"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model1(input_ids=input_ids, attention_mask=attention_mask).logits
            preds1 += logits.squeeze(-1).tolist()
    # free GPU
    del model1
    torch.cuda.empty_cache()

    # ========== 4) Run Oracle #2 ==========
    oracle2_ckpt = config["paths"]["oracle_path2"]
    oracle2_weights = config["paths"]["checkpoint_path2"]
    _, model2 = load_model(
        oracle2_ckpt, oracle2_weights, num_labels=1
    )
    model2.to(device).eval()
    preds2 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle2"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model2(input_ids=input_ids, attention_mask=attention_mask).logits
            preds2 += logits.squeeze(-1).tolist()

    # ========== 5) Build DataFrame & Save ==========
    df = pd.DataFrame({
        "name":        [r["name"]       for r in records],
        "index":       [r["index"]      for r in records],
        "prediction1": preds1,
        "prediction2": preds2,
        "mean_prediction": [np.mean([preds1[i], preds2[i]]) for i in range(len(preds1))]
    })

    output_path = config["paths"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")

if __name__ == "__main__":
    main()


