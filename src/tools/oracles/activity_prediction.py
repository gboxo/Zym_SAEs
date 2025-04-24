import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
import pickle as pkl
from peft import LoraConfig, inject_adapter_in_model
from datasets import Dataset
from src.tools.oracles.oracles_utils import load_config

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


def save_model(model, filepath):
    non_frozen_params = {
        param_name: param
        for param_name, param in model.named_parameters() if param.requires_grad
    }
    torch.save(non_frozen_params, filepath)


def load_model(checkpoint, filepath, num_labels=1, mixed=False, full=False, deepspeed=True):
    model, tokenizer = (
        load_esm_model(checkpoint, num_labels, mixed, full, deepspeed)
        if "esm" in checkpoint
        else load_T5_model(checkpoint, num_labels, mixed, full, deepspeed)
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



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--iteration_num", type=int, required=True)

    args = parser.parse_args()
    cfg_path = args.cfg_path
    iteration_num = args.iteration_num
    config = load_config(cfg_path)
    ec_label = config["label"]


    seqs_path = config["paths"]["seqs_path"].format(ec_label, iteration_num)
    output_path = config["paths"]["output_path"].format(iteration_num)



    print(f"Loading the Oracle model")

    checkpoint = config["paths"]["oracle_path1"]
    tokenizer, model = load_model(
        checkpoint,
        config["paths"]["checkpoint_path1"],
        num_labels=1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Generating the dataset")


    test_dataloader, names = generate_dataset(seqs_path, tokenizer)



    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions.extend(logits.squeeze().tolist())

    del model
    torch.cuda.empty_cache()

    checkpoint = config["paths"]["oracle_path2"]
    tokenizer, model = load_model(
        checkpoint,
        config["paths"]["checkpoint_path2"],
        num_labels=1
    )
    model.to(device)
    model.eval()

    predictions2 = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            predictions2.extend(logits.squeeze().tolist())



    # Write CSV header
    out = "name,prediction1,prediction2\n"
    
    # Add data rows
    for name, prediction1, prediction2 in zip(names, predictions, predictions2):
        name = name.split("\t")[0]
        out += f"{name},{prediction1},{prediction2}\n"
        
    print("Saving the predictions on ", output_path)

    with open(output_path, 'w') as f:
        f.write(out)

    del model
    torch.cuda.empty_cache()