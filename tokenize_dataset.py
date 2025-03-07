# %%
import pandas as pd
import torch
import numpy as np
from src.utils import load_config, load_model, get_paths
# %%
paths = get_paths()
model_path = paths.model_path
tokenizer, model = load_model(model_path)

# %%
from datasets import load_from_disk
dataset_path = "Data/Diffing/dataset_iteration1"
dataset = load_from_disk(dataset_path)
train_dataset = dataset["train"]
eval_dataset = dataset["eval"]
sequences_train = train_dataset["sequence"]
sequences_eval = eval_dataset["sequence"]

# %%

tokenized_train = [tokenizer.encode("4.2.1.1<sep><start>"+seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_train]
tokenized_eval = [tokenizer.encode("4.2.1.1<sep><start>"+seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_eval]

# %%
tokenized_train = np.array(tokenized_train).tolist()
tokenized_eval = np.array(tokenized_eval).tolist()

# %%
# Create a new dataset with the tokenized sequences just 1 column input_ids
# The current mapping approach has an issue - x is an index, not the actual data
# Let's fix this by using enumerate and creating a proper mapping

# Create new datasets with just the input_ids column
from datasets import Dataset

# Convert tokenized sequences to datasets with just input_ids
train_tokenized_dataset = Dataset.from_dict({"input_ids": tokenized_train})
eval_tokenized_dataset = Dataset.from_dict({"input_ids": tokenized_eval})

# Save the tokenized datasets to disk
train_tokenized_dataset.save_to_disk("Data/Diffing/tokenized_train_dataset_iteration1")
eval_tokenized_dataset.save_to_disk("Data/Diffing/tokenized_eval_dataset_iteration1")

# %%


# ============= DATASET WITH ALL SEQUENCES GENERATED IN ROUNDS 0 TO 10 =============

# ================== LOAD ALL FASTA FILES ==================


import os
RL_data_path = "Data/Diffing/RL_GEN_Sequences"
files = os.listdir(RL_data_path)
files = [x for x in files if x.endswith(".fasta")]
fasta_dict = {}
for file in files:
    with open(RL_data_path+"/"+file,"r") as f:
        fasta_data = f.read()
        fasta_data = fasta_data.split(">")

    for x in fasta_data:
        if x == "":
            continue
        x = x.split("\n")
        d = {
                "id":x[0].split("\t")[0],
                "score":x[0].split("\t")[1],
                "sequence":x[1],

                }
        fasta_dict[d["id"]] = d

seqs = [x["sequence"] for x in fasta_dict.values()]
ids = [x["id"] for x in fasta_dict.values()]


# Train/val split

indices = list(range(len(seqs)))
np.random.shuffle(indices)
train_indices = indices[:int(0.8*len(indices))]
val_indices = indices[int(0.8*len(indices)):]


train_seqs = ["4.2.1.1<sep><start>"+seqs[i] for i in train_indices]
val_seqs = ["4.2.1.1<sep><start>"+seqs[i] for i in val_indices]
train_ids = [ids[i] for i in train_indices]
val_ids = [ids[i] for i in val_indices]

train_seqs_tokenized = [tokenizer.encode(seq, padding="max_length", truncation=True, max_length=256) for seq in train_seqs]
val_seqs_tokenized = [tokenizer.encode(seq, padding="max_length", truncation=True, max_length=256) for seq in val_seqs]







from datasets import Dataset

# Convert tokenized sequences to datasets with just input_ids
train_tokenized_dataset = Dataset.from_dict({"input_ids": train_seqs_tokenized})
eval_tokenized_dataset = Dataset.from_dict({"input_ids": val_seqs_tokenized})

# Save the tokenized datasets to disk
train_tokenized_dataset.save_to_disk("Data/Diffing/tokenized_train_dataset_iteration1_rounds0to10")
eval_tokenized_dataset.save_to_disk("Data/Diffing/tokenized_eval_dataset_iteration1_rounds0to10")















