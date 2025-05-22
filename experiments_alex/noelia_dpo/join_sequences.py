from datasets import load_from_disk, concatenate_datasets, DatasetDict
from src.utils import load_model
import pandas as pd
import os
import argparse
from datasets import concatenate_datasets
from tqdm import tqdm
from datasets import Dataset
import random
import hashlib


"""
This implementation is different from the standard way because we filtered the sequences by the tmscore.

So we'll be loading the sequences from the dataframes and then we'll be filtering them by the tmscore.


"""



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
    return all_tokens

def read_fasta(fasta_path):
    with open(fasta_path, "r") as f:
        sequences = f.read()
    data = sequences.split(">")
    sequences = []
    for d in data:
        if len(d) > 0:
            d = d.split("\n")
            sequence = d[1].strip()
            sequences.append(sequence)
    return list(set(sequences))



def get_df_and_join_datasets(sequences_path, out_path, tokenizer):

    all_sequences = {}


    all_sequences_folders = os.listdir(sequences_path)
    for folder in all_sequences_folders:
        seq_path = os.path.join(sequences_path, folder)
        files = os.listdir(seq_path)
        if len(files) != 1:
            continue
        file = files[0]
        seq_path = os.path.join(seq_path, file)
        sequences = read_fasta(seq_path)
        all_sequences[folder] = sequences
    
    # sort the dict by key (model_i)
    all_sequences = dict(sorted(all_sequences.items(), key=lambda x: int(x[0].split("_")[1])))
    for key, sequences in all_sequences.items():
        print(key, len(set(sequences)))
    



    for folder, sequences in all_sequences.items():

        tokens = get_all_tokens(tokenizer, sequences)
        random_indices = random.sample(range(len(tokens)), int(len(tokens)*0.8))
        train_tokens = [tokens[i] for i in random_indices]
        test_tokens = [tokens[i] for i in range(len(tokens)) if i not in random_indices]
        train_dict = {"input_ids": train_tokens}
        test_dict = {"input_ids": test_tokens}
        final_dataset = DatasetDict({
            "train": Dataset.from_dict(train_dict),
            "eval": Dataset.from_dict(test_dict)
        })
        final_dataset.save_to_disk(f"{out_path}/dataset_{folder}")


    





if __name__ == "__main__":

    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"



    tokenizer, model = load_model(model_path)
    del model
    sequences_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/generated_seqs_by_model/"
    out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/"
    os.makedirs(out_path, exist_ok=True)


    get_df_and_join_datasets(sequences_path, out_path, tokenizer)