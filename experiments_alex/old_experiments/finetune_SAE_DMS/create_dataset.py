from transformers import AutoTokenizer
import os
import pandas as pd
from src.utils import load_model
import torch
from tqdm import tqdm
from datasets import Dataset, DatasetDict
# %%


seed = 2003



def get_data():
    # The first line is the header
    data = pd.read_csv("/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/alpha-amylase-training-data.csv",header=0)

    # The first column is the sequence
    return data

def get_tokens(tokenizer, sequence):
    sequence = str(f"3.2.1.1<sep><start>{sequence}<|endoftext|>")
    tokens = tokenizer.encode(sequence, max_length=512, padding="max_length", truncation=True)
    return tokens


def get_all_tokens(tokenizer, sequences):
    all_tokens = []
    for sequence in tqdm(sequences):
        inputs = get_tokens(tokenizer, sequence)
        all_tokens.append(inputs)
        del inputs
        torch.cuda.empty_cache()
    return all_tokens

    
    
    

def main(tokenizer):
    df = get_data()
    sequences = df["mutated_sequence"].values
    tokens = get_all_tokens(tokenizer, sequences)
    # Save results to file
    df = pd.DataFrame({"input_ids": tokens})
    hf_dataset = Dataset.from_pandas(df)
    shuffled_dataset = hf_dataset.shuffle(seed=seed)
    # Split the dataset (80% train, 20% eval)
    split_percent = 0.2
    train_size = int((1-split_percent) * len(shuffled_dataset))
    train_dataset = shuffled_dataset.select(range(train_size))
    eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))

    final_dataset = DatasetDict({
        'train': train_dataset,
        'eval': eval_dataset
        })
    os.makedirs("/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/dataset", exist_ok=True)    
    out_path = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/dataset"
    os.makedirs(out_path, exist_ok=True)    
    final_dataset.save_to_disk(out_path)

if __name__ == "__main__":
    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"



    tokenizer, model = load_model(model_path)
    del model
    all_tokens = main(tokenizer)

