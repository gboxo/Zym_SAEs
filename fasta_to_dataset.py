# %%
import pandas as pd
import torch
import numpy as np
from src.utils import load_model, get_paths
import re
import os
from datasets import Dataset, DatasetDict, Value
# %%
paths = get_paths()
model_path = paths.model_path
tokenizer, model = load_model(model_path)


# %%

paths = "/users/nferruz/gboxo/Alpha Amylase/seq_gens/"
files = os.listdir(paths)
files = [file for file in files if file.endswith('.fasta')]

iterations = [int(re.findall(r'\d+', file)[-1]) for file in files]
arg_sort = np.argsort(iterations)
sorted_files = [files[i] for i in arg_sort]

# %%
def fasta_to_dataset(file):
    with open(paths+file, 'r') as f:
        data = f.read()
        data = data.split('>')
        data = [elem for elem in data if elem != '']
        data = [elem.split('\n')[1] for elem in data]

        ids = np.arange(len(data))
        shuffle_ids = np.random.permutation(ids)
        train_ids = shuffle_ids[:int(len(data)*0.8)]
        eval_ids = shuffle_ids[int(len(data)*0.8):]

        sequences_train = [data[i] for i in train_ids]
        sequences_eval = [data[i] for i in eval_ids]

        tokenized_train = [tokenizer.encode("3.2.1.1<sep><start>"+seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_train]
        tokenized_eval = [tokenizer.encode("3.2.1.1<sep><start>"+seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_eval]
        tokenized_train = np.array(tokenized_train).tolist()
        tokenized_eval = np.array(tokenized_eval).tolist()

        df_train = pd.DataFrame({"input_ids": tokenized_train})
        df_eval = pd.DataFrame({"input_ids": tokenized_eval})

        train_dataset = Dataset.from_pandas(df_train.rename(columns={"input_ids": "input_ids"}), split="train")
        eval_dataset = Dataset.from_pandas(df_eval.rename(columns={"input_ids": "input_ids"}), split="eval")


        return train_dataset, eval_dataset
        

# %%


for i in range(len(sorted_files)):
    train_dataset, eval_dataset = fasta_to_dataset(sorted_files[i])
    # Save the datasets
    train_dataset.save_to_disk("Data/Diffing Alpha Amylase/tokenized_train_dataset_iteration"+str(i))
    eval_dataset.save_to_disk("Data/Diffing Alpha Amylase/tokenized_eval_dataset_iteration"+str(i))

# %%
