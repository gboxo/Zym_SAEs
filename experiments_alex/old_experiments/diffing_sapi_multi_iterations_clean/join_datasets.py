from datasets import load_from_disk, concatenate_datasets, DatasetDict
from src.utils import load_model
import pandas as pd
import os
import argparse
from src.tools.data_utils.data_utils import load_config
from datasets import concatenate_datasets
from tqdm import tqdm
from datasets import Dataset
import random



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





def get_df_and_join_datasets(dataframes_path, out_path):

    
    dataframes = [dataframes_path+f"dataframe_all_iteration{i}.csv" for i in range(6)]
    for idx, dataframe in enumerate(dataframes):
        df = pd.read_csv(dataframe)
        sequences = df["sequence"].tolist()
        

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
        final_dataset.save_to_disk(f"{out_path}/dataset_iteration{idx}")


    





if __name__ == "__main__":

    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"



    tokenizer, model = load_model(model_path)
    del model
    dataframes_path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/joined_dataframes/"
    out_path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/"


    get_df_and_join_datasets(dataframes_path, out_path)