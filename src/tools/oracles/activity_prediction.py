import torch
import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
from src.tools.oracles.oracles_utils import load_model, read_sequence_from_file, build_dataloader_from_records
from src.tools.oracles.activity_prediction_config import ActivityPredictionConfig

def run_predictions(config: ActivityPredictionConfig):
    # Load sequences
    records = []
    if os.path.isdir(config.seq_source):
        for fn in sorted(os.listdir(config.seq_source)):
            if not fn.endswith(".txt"): continue
            full_path = os.path.join(config.seq_source, fn)
            seqs_dict = read_sequence_from_file(full_path)
            base = os.path.splitext(fn)[0]
            for seq_id, seq in seqs_dict.items():
                records.append({"name": base, "index": seq_id, "sequence": seq})
    else:
        seqs_dict = read_sequence_from_file(config.seq_source)
        base = os.path.splitext(os.path.basename(config.seq_source))[0]
        for seq_id, seq in seqs_dict.items():
            records.append({"name": base, "index": seq_id, "sequence": seq})

    # Setup
    tokenizer, _ = load_model(config.oracle_path1, config.checkpoint_path1, num_labels=1)
    loader = build_dataloader_from_records(records, tokenizer, config.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run both oracles
    def run_oracle(oracle_path, checkpoint_path, desc):
        _, model = load_model(oracle_path, checkpoint_path, num_labels=1)
        model.to(device).eval()
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                preds += logits.squeeze(-1).tolist()
        del model
        torch.cuda.empty_cache()
        return preds

    preds1 = run_oracle(config.oracle_path1, config.checkpoint_path1, "Oracle1")
    preds2 = run_oracle(config.oracle_path2, config.checkpoint_path2, "Oracle2")

    # Save results
    df = pd.DataFrame({
        "name": [r["name"] for r in records],
        "index": [r["index"] for r in records],
        "sequence": [r["sequence"] for r in records],
        "prediction1": preds1,
        "prediction2": preds2,
        "mean_prediction": [np.mean([p1, p2]) for p1, p2 in zip(preds1, preds2)]
    })
    
    os.makedirs(os.path.dirname(config.out_dir), exist_ok=True)
    df.to_csv(config.out_dir, index=False)
    print(f"Saved predictions → {config.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    
    # Simple one-liner to get unified config
    config = ActivityPredictionConfig.from_yaml(args.cfg_path, batch_size=args.batch_size)
    run_predictions(config)
