import torch
from transformers import AutoTokenizer
import numpy as np
import os
import random
import argparse
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict

seed = 1998


def generate_dataset(iteration_num, ec_label):
    data = dict()
    path = f"/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_Clean/DPO_clean_amylase_run_SAPI_only_gerard/seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    path = f"/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_{ec_label}_iteration{iteration_num}.fasta"
    with open(path, "r") as f:
        rep_seq = f.readlines()

    sequences_rep = dict()
    
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").replace("\n", "")
            emb_identifier = line.replace(">", "").replace("\n", "")
        else:
            aa = line.strip()
            sequences_rep[name] = {
                            "sequence" : aa,
                            "emb_identifier" : emb_identifier
                                    }
    

    data["input_ids"] = [formatting_sequence(elem["sequence"], ec_label) for elem in sequences_rep.values()]
    df = pd.DataFrame(data)
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
    
        
    final_dataset.save_to_disk(f"/home/woody/b114cb/b114cb23/boxo/diffing_datasets_big/dataset_iteration{iteration_num}")
    
    return final_dataset
     

def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<|endoftext|>")
    tokens = tokenizer.encode(sequence, max_length=512, padding="max_length", truncation=True)
    return tokens


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label
    ec_label = ec_label.strip()
    seed_everything(seed)
    

    model_name = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
        
 
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    if not os.path.exists(f"dataset_iteration{iteration_num}"):
      dataset = generate_dataset(iteration_num, ec_label)
    else:
      dataset = load_from_disk(f"dataset_iteration{iteration_num}")
    








