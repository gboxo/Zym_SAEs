from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os
import argparse
from src.tools.data_utils.data_utils import load_config


def get_and_join_datasets(datasets_path, out_path):

    folders = os.listdir(datasets_path)


    all_datasets_train = []
    all_datasets_eval = []
    for folder in folders:
        dataset_train = load_from_disk(f"{datasets_path}/{folder}/train")
        dataset_eval = load_from_disk(f"{datasets_path}/{folder}/eval")
        all_datasets_train.append(dataset_train)
        all_datasets_eval.append(dataset_eval)

    final_dataset_train = concatenate_datasets(all_datasets_train)
    final_dataset_eval = concatenate_datasets(all_datasets_eval)


    final_dataset = DatasetDict({
        "train": final_dataset_train,
        "eval": final_dataset_eval
    })
    final_dataset.save_to_disk(out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)
    get_and_join_datasets(config["paths"]["datasets_path"], config["paths"]["out_path"])
