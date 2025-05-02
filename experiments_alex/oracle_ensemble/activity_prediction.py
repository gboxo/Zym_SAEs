import os
import argparse
from tqdm import tqdm
import torch
import pandas as pd

from src.tools.oracles.oracles_utils import load_config
from src.tools.oracles.activity_prediction import load_model, SequenceDataset
from torch.utils.data import DataLoader





def main():

    df_path = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
    oracle_path1 = "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
    checkpoint_path1 = "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth"

    oracle_path2 = "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
    checkpoint_path2 = "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth"
    out_path = "/home/woody/b114cb/b114cb23/boxo/oracle_ensemble/activity_predictions.csv"

    

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_csv(df_path)

    sequences = df["mutated_sequence"].tolist()
    names = df["mutant"].tolist()
    



    # Load config (to get oracle checkpoint paths)

    # ========== 1) Read all sequences (flatten multiple entries per file) ==========
    records = []
    for idx_seq, (name, seq) in enumerate(zip(names, sequences)):
        records.append({
            "name":      name,
            "index":     idx_seq,   # per‐file sequence index
            "sequence":  seq
        })

    # ========== 2) Tokenize & build DataLoader ==========
    tokenizer, _ = load_model(
        oracle_path1, checkpoint_path1, num_labels=1
    )
    # Tokenize all sequences in a batch
    tokenized = tokenizer(
        sequences,
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
    loader = DataLoader(tensor_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========== 3) Run Oracle #1 ==========
    _, model1 = load_model(
        oracle_path1, checkpoint_path1, num_labels=1
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
    _, model2 = load_model(
        oracle_path2, checkpoint_path2, num_labels=1
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
        "name":        names,
        "index":       [r["index"]      for r in records],
        "prediction1": preds1,
        "prediction2": preds2
    })

    df.to_csv(out_path, index=False)
    print(f"Saved predictions → {out_path}")

if __name__ == "__main__":
    main()


