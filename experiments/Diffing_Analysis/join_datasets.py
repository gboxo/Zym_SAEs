from datasets import load_from_disk, concatenate_datasets, DatasetDict
import os

path = "/home/woody/b114cb/b114cb23/boxo/diffing_datasets_0_30/"
folders = os.listdir(path)


all_datasets_train = []
all_datasets_eval = []
for folder in folders:
    dataset_train = load_from_disk(f"{path}/{folder}/train")
    dataset_eval = load_from_disk(f"{path}/{folder}/eval")
    all_datasets_train.append(dataset_train)
    all_datasets_eval.append(dataset_eval)

final_dataset_train = concatenate_datasets(all_datasets_train)
final_dataset_eval = concatenate_datasets(all_datasets_eval)


final_dataset = DatasetDict({
    "train": final_dataset_train,
    "eval": final_dataset_eval
})

print(final_dataset)
final_dataset.save_to_disk("/home/woody/b114cb/b114cb23/boxo/diffing_datasets_0_30/final_dataset_iteration_0_30")
