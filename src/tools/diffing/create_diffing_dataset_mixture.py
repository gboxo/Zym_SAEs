from transformers import AutoTokenizer
import os
import random
import argparse
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict

seed = 1998


def generate_dataset(iteration_num, ec_label, mixture_ratio=0.8):
    data = dict()
    
    # Load sequences from fasta file
    #fasta_path = "/home/woody/b114cb/b114cb23/boxo/seq_gens/seq_gen_amylase_iteration1.fasta"
    fasta_path = f"/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_Clean/DPO_clean_amylase_run_SAPI_only_gerard/seq_gen_3.2.1.1_iteration{iteration_num}.fasta"

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
    
    # Get sequences from fasta file
    fasta_input_ids = [formatting_sequence(elem["sequence"], ec_label) for elem in sequences_rep.values()]
    print("Length of fasta_input_ids: ", len(fasta_input_ids))
    rest = int((len(fasta_input_ids) - int(len(fasta_input_ids) * mixture_ratio))/mixture_ratio)
    print("Rest: ", rest)
    
    # Load existing tokenized dataset
    existing_dataset = load_from_disk("/home/woody/b114cb/b114cb23/boxo/new_dataset_concat_train/")
    existing_dataset = existing_dataset.shuffle(seed=seed)

    existing_dataset = existing_dataset[:10000]
    # Shuffle the existing dataset

    existing_input_ids = existing_dataset['input_ids']

    
    # Randomly sample from existing dataset
    print("Taking a sample of ", rest, " sequences from existing dataset")
    sampled_existing = existing_input_ids[:rest]
    # Combine both sources
    print("Combining both sources")
    data["input_ids"] = fasta_input_ids + sampled_existing

    # Continue with the existing code
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
    
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/diffing_datasets_mixture/dataset_iteration{iteration_num}", exist_ok=True)
    
    final_dataset.save_to_disk(f"/home/woody/b114cb/b114cb23/boxo/diffing_datasets_mixture/dataset_iteration{iteration_num}")
    
    return final_dataset
     


def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<|endoftext|>")
    tokens = tokenizer.encode(sequence, max_length=512, padding="max_length", truncation=True)
    return tokens


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    parser.add_argument("--mixture_ratio", type=float, default=0.8, 
                       help="Ratio of sequences from fasta file (default: 0.8)")
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label.strip()
    mixture_ratio = args.mixture_ratio
    
    model_name = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    print(tokenizer)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    dataset = generate_dataset(iteration_num, ec_label, mixture_ratio)
    








