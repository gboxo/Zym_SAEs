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

tokenized_train = [tokenizer.encode(seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_train]
tokenized_eval = [tokenizer.encode(seq, max_length=256, truncation=True, padding="max_length") for seq in sequences_eval]

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










