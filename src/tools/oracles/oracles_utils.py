import yaml

import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
from peft import LoraConfig, inject_adapter_in_model
from datasets import Dataset
from src.tools.oracles.oracles_utils import load_config
import numpy as np
import pandas as pd

def load_config(cfg_path):
    with open(cfg_path, 'r') as f:
        config = yaml.safe_load(f)
    return config





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

def load_esm_model(checkpoint, num_labels, half_precision, full=False, deepspeed=True):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels,
        torch_dtype=torch.float16 if half_precision and deepspeed else None
    )
    if full:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )
    model = inject_adapter_in_model(peft_config, model)
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True
    return model, tokenizer




def load_model(checkpoint, filepath, num_labels=1, mixed=False, full=False, deepspeed=True):
    model, tokenizer = (
        load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
    )
    non_frozen_params = torch.load(filepath)
    for param_name, param in model.named_parameters():
        if param_name in non_frozen_params:
            param.data = non_frozen_params[param_name].data
    return tokenizer, model


def generate_dataset(seq_path, tokenizer):
    tokenized_sequences = []
    names = []
    with open(seq_path, "r") as f:
        rep_seq = f.readlines()
    for line in rep_seq:
        line = line.replace("\n","")
        sections  = line.split(",")
        seq = sections[1]
        seq = seq.replace("3. 2. 1. 1 <sep> <start>","").replace("<end>","")
        
        if not line.startswith(">"):
            seq = line.strip()
            encoded = tokenizer(
                seq, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokenized_sequences.append(encoded)
        else:
            names.append(line.split(" ")[0])

    dataset = SequenceDataset(tokenized_sequences)
    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    return test_dataloader, names


# helper: read FASTA-style sequences into a dict {id: seq}
def read_sequence_from_file(path):
    seqs = {}
    curr = ""
    curr_id = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr and curr_id:
                    seqs[curr_id] = curr
                    curr = ""
                curr_id = line[1:].split(",")[0]
                if "<start>" in line and "<end>" in line:
                    block = line.split("<start>",1)[1].split("<end>",1)[0]
                    curr = block.replace(" ","")
                continue
            curr += line.replace(" ","")
    if curr and curr_id:
        seqs[curr_id] = curr
    return seqs


# helper: build a DataLoader from a list of sequence records
# records: List of dicts with keys 'sequence', optionally 'name'/'index'
def build_dataloader_from_records(records, tokenizer, batch_size):
    """
    Tokenizes sequences and returns a torch DataLoader.
    """
    # Batch tokenization
    tokenized = tokenizer(
        [r['sequence'] for r in records],
        max_length=1024,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    # Wrap each sequence as a dict for SequenceDataset
    tokenized_seqs = [
        { 'input_ids': tokenized['input_ids'][i:i+1], 'attention_mask': tokenized['attention_mask'][i:i+1] }
        for i in range(len(records))
    ]
    dataset = SequenceDataset(tokenized_seqs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for KL workflow")

    args = parser.parse_args()
    cfg_path = args.cfg_path
    batch_size = args.batch_size
    config = load_config(cfg_path)
    # reading all sequences from a file or directory
    seq_source = config["paths"]["seqs_path"]
    records = []
    if os.path.isdir(seq_source):
        # iterate over all .txt files in directory
        for fn in sorted(os.listdir(seq_source)):
            if not fn.endswith(".txt"): continue
            full_path = os.path.join(seq_source, fn)
            seqs_dict = read_sequence_from_file(full_path)
            base = os.path.splitext(fn)[0]
            for seq_id, seq in seqs_dict.items():
                records.append({"name": base, "index": seq_id, "sequence": seq})
    else:
        # single-file input
        seqs_dict = read_sequence_from_file(seq_source)
        base = os.path.splitext(os.path.basename(seq_source))[0]
        for seq_id, seq in seqs_dict.items():
            records.append({"name": base, "index": seq_id, "sequence": seq})
    # Tokenize and build DataLoader
    tokenizer, _ = load_model(
        config["paths"]["oracle_path1"],
        config["paths"]["checkpoint_path1"],
        num_labels=1
    )
    # Build DataLoader using helper
    loader = build_dataloader_from_records(records, tokenizer, batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run Oracle #1
    _, model1 = load_model(
        config["paths"]["oracle_path1"],
        config["paths"]["checkpoint_path1"],
        num_labels=1
    )
    model1.to(device).eval()
    preds1 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle1"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model1(input_ids=input_ids, attention_mask=attention_mask).logits
            preds1 += logits.squeeze(-1).tolist()
    del model1
    torch.cuda.empty_cache()
    # Run Oracle #2
    _, model2 = load_model(
        config["paths"]["oracle_path2"],
        config["paths"]["checkpoint_path2"],
        num_labels=1
    )
    model2.to(device).eval()
    preds2 = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Oracle2"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model2(input_ids=input_ids, attention_mask=attention_mask).logits
            preds2 += logits.squeeze(-1).tolist()
    del model2
    torch.cuda.empty_cache()
    # Build DataFrame and save
    df = pd.DataFrame({
        "name": [r["name"] for r in records],
        "index": [r["index"] for r in records],
        "prediction1": preds1,
        "prediction2": preds2,
        "mean_prediction": [np.mean([preds1[i], preds2[i]]) for i in range(len(preds1))]
    })
    output_path = config["paths"]["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved predictions → {output_path}")


