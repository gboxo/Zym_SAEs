import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd

from src.tools.oracles.oracles_utils import load_config
from src.tools.oracles.activity_prediction import load_model, SequenceDataset
from torch.utils.data import DataLoader




def read_sequence_from_file(path):
    """
    Reads a .txt file in which each FASTA header looks like:
      >3.2.1.1_0 <sep> <start> A B C D … <end>
    This function will:
      – find the substring between '<start>' and '<end>'
      – strip out all spaces
      – return a single concatenated sequence string
    """
    seq_parts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # header + seq on one line
                if '<start>' in line and '<end>' in line:
                    # grab what's between <start> and <end>
                    seq_block = line.split('<start>', 1)[1].split('<end>', 1)[0]
                    seq_block = seq_block.replace(' ', '')
                    seq_parts.append(seq_block)
                # else: no inline sequence, skip header
                continue

            # if the file ever has sequence on its own line(s),
            # remove spaces (assuming tokens separated by spaces)
            seq_parts.append(line.replace(' ', ''))

    # join all fragments (if multiple entries per file, they’ll concatenate)
    return seq_parts

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

    # ========== 1) Read all sequences ==========
    records = []
    for fn in os.listdir(config["paths"]["seqs_path"]):
        if not fn.endswith(".txt"):
            continue
        full = os.path.join(config["paths"]["seqs_path"], fn)
        seq = read_sequence_from_file(full)
        records.append({
            "name": os.path.splitext(fn)[0],
            "sequence": seq[0]
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
        "name": [r["name"] for r in records],
        "prediction1": preds1,
        "prediction2": preds2
    })

    output_path = config["paths"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")

if __name__ == "__main__":
    main()


