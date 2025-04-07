import torch
from transformers import AutoTokenizer
import numpy as np
import os
import random
import argparse
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
from src.tools.data_utils.data_utils import load_config


seed = 1998


def generate_dataset(fasta_path, out_path, tokenizer):
    data = dict()
    with open(fasta_path, "r") as f:
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
    
    os.makedirs(out_path, exist_ok=True)    
    final_dataset.save_to_disk(out_path)
    
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
    parser.add_argument("--cfg_path", type=str)
    parser.add_argument("--iteration_num", type=int)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    seed_everything(seed)
    ec_label = "3.2.1.1"
    config = load_config(cfg_path)
    iteration_num = args.iteration_num
    fasta_path = config["paths"]["fasta_path"].format(ec_label=ec_label, iteration_num=iteration_num)
    out_path = config["paths"]["out_path"].format(iteration_num=iteration_num)

    model_name = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
        
 
    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    dataset = generate_dataset(fasta_path, out_path, tokenizer)
    








