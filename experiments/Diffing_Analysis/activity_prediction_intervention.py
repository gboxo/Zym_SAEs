import torch
import os
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse
import pickle as pkl
from peft import LoraConfig, inject_adapter_in_model
from datasets import Dataset


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
        if not line.startswith(">"):
            encoded = tokenizer(
                line.strip(), max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
            )
            tokenized_sequences.append(encoded)
        else:
            names.append(line)
    dataset = SequenceDataset(tokenized_sequences)
    test_dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    return test_dataloader, names



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--procedure", type=str, required=True)
    args = parser.parse_args()

    iteration_num = args.iteration_num
    ec_label = args.label.strip()
    procedure = args.procedure # [ "steering", "ablation"]
    path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/correlations/top_correlations_M{iteration_num}_D{iteration_num}.pkl"
    with open(path, "rb") as f:
        top_correlations = pkl.load(f)
    feature_indices = top_correlations["feature_indices"]
    all_seqs_paths = []
    for feature in feature_indices:
        seqs_path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/{procedure}/M{iteration_num}_D{iteration_num}/{procedure}_feature_{feature}.txt"
        all_seqs_paths.append(seqs_path)
    output_path = f"/home/woody/b114cb/b114cb23/boxo/activity_predictions_{procedure}/activity_prediction_iteration{iteration_num}_feature_{feature}.txt"
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/activity_predictions_{procedure}", exist_ok=True)







    print(f"Loading the Oracle model")

    checkpoint = "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D/"
    tokenizer, model = load_model(
        checkpoint,
        "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth",
        num_labels=1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    print(f"Generating the dataset")

    all_predictions1 = {} 

    for i,seq_path in enumerate(all_seqs_paths):
        feature = feature_indices[i]
        all_predictions1[feature] = {} 
        test_dataloader, names = generate_dataset(seq_path, tokenizer)

        print(f"Generating the predictions")
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                predictions.extend(logits.squeeze().tolist())
        for name, prediction in zip(names, predictions):
            all_predictions1[feature][name] = prediction


    del model
    torch.cuda.empty_cache()

    checkpoint = "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
    tokenizer, model = load_model(
        checkpoint,
        "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth",
        num_labels=1
    )
    model.to(device)
    model.eval()

    all_predictions2 = {} 

    for i,seq_path in enumerate(all_seqs_paths):
        feature = feature_indices[i]
        all_predictions2[feature] = {} 
        test_dataloader, names = generate_dataset(seq_path, tokenizer)

        print(f"Generating the predictions")
        predictions2 = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                predictions2.extend(logits.squeeze().tolist())
        for name, prediction in zip(names, predictions2):
            all_predictions2[feature][name] = prediction

    all_predictions = {} 
    for feature in all_predictions1:
        all_predictions[feature] = {} 
        for name in all_predictions1[feature]:
            all_predictions[feature][name] = (all_predictions1[feature][name] + all_predictions2[feature][name]) / 2


    out = "".join(f'{name},{feature},{prediction}\n' for feature, prediction in all_predictions.items() for name, prediction in prediction.items())
    os.makedirs(f'/home/woody/b114cb/b114cb23/boxo/activity_predictions_{procedure}', exist_ok=True)
    with open(f'/home/woody/b114cb/b114cb23/boxo/activity_predictions_{procedure}/activity_prediction_iteration{iteration_num}.txt', 'w') as f:
        f.write(out)

    del model
    torch.cuda.empty_cache()