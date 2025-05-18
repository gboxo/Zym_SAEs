from datasets import load_from_disk, concatenate_datasets, DatasetDict
from src.utils import load_model
import os
import argparse
from src.tools.data_utils.data_utils import load_config
from datasets import concatenate_datasets
from tqdm import tqdm
from datasets import Dataset




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





def get_and_join_datasets(datasets_path, out_path):

    folders = [[f"dataset_iteration{j}" for j in range(i,i+5)] for i in range(1,30,5)]
    print(folders)


    for idx,folders in enumerate(folders):
        all_datasets_train = []
        all_datasets_eval = []
        for folder in folders:
            dataset_train = load_from_disk(f"{datasets_path}/{folder}/train")
            dataset_eval = load_from_disk(f"{datasets_path}/{folder}/eval")
            train_sequences = dataset_train["sequence"]
            test_sequences = dataset_eval["sequence"]
            train_tokens = get_all_tokens(tokenizer, train_sequences)
            test_tokens = get_all_tokens(tokenizer, test_sequences)
            train_dict = {"input_ids": train_tokens}
            test_dict = {"input_ids": test_tokens}
            dataset_train = Dataset.from_dict(train_dict)
            dataset_eval = Dataset.from_dict(test_dict)
            all_datasets_train.append(dataset_train)
            all_datasets_eval.append(dataset_eval)
        final_dataset_train = concatenate_datasets(all_datasets_train)
        final_dataset_eval = concatenate_datasets(all_datasets_eval)
        final_dataset = DatasetDict({
            "train": final_dataset_train,
            "eval": final_dataset_eval
        })
        final_dataset.save_to_disk(f"{out_path}/dataset_iteration{idx}")

if __name__ == "__main__":

    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"



    tokenizer, model = load_model(model_path)
    del model
    datasets_path = "/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/"
    os.makedirs(datasets_path, exist_ok=True)
    out_path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations/"


    get_and_join_datasets(datasets_path, out_path)
