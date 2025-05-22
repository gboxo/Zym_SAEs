import pandas as pd
import os
from collections import Counter
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
import pickle
import torch


"""
Position,Count,Avg KL,Same Token %,Top Transitions
285,153,5.889991454439225,28.75816993464052,"['449 (S) → 442 (L) (103 times)']
293,136,2.318561604794334,2.2058823529411766,"['440 (I) → 442 (L) (103 times)' ]
107,117,2.886726530189188,3.418803418803419,"['440 (I) → 437 (F) (112 times)']
102,88,0.9261001914062283,23.863636363636363,['440 (I) → 442 (L) (67 times)']
138,84,1.4521822613619624,9.523809523809524,['449 (S) → 439 (H) (76 times)']

"""


kl_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_kl_divergence.pkl"

dataset = load_from_disk("/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_3/eval/")



tokenizer = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/model-3.2.1.1/")

out_path = "/home/woody/b114cb/b114cb23/boxo/kl_divergence/trans_comb/"
os.makedirs(out_path, exist_ok=True)

input_ids = dataset['input_ids']

seqs = [tokenizer.decode(seq) for seq in input_ids]

special_tokens = ["3.2.1.1<sep><start>", "<sep>", "<start>", "<end>", "<pad>", " "]

# Remove the special tokens from the sequences
seqs = [seq.replace("<sep>","") for seq in seqs]
seqs = [seq.replace("<start>","") for seq in seqs]
seqs = [seq.replace("<end>","") for seq in seqs]
seqs = [seq.replace("<pad>","") for seq in seqs]
seqs = [seq.replace("<|endoftext|>","") for seq in seqs]
seqs = [seq.replace(" ","") for seq in seqs]
seqs = [seq.replace("3.2.1.1","") for seq in seqs]

positions = [285, 293, 107, 102, 138]
positions = [pos-8 for pos in positions]
position_transition_dict = {
    277:("L", "S"),
    285:("L", "I"),
    99:("F", "I"),
    94:("L", "I"),
    130:("H", "S"),
}

feat_dict_orig = {}
feat_dict_trans = {}
for pos in positions:
    orig_l = []
    trans_l = []
    aa_counts = Counter([seq[pos] for seq in seqs])
    for seq in seqs:
        if seq[pos] == position_transition_dict[pos][0]:
            orig_l.append(seq)
            seq_trans = seq[:pos] + position_transition_dict[pos][1] + seq[pos+1:]
            trans_l.append(seq_trans)

    feat_dict_orig[pos] = orig_l
    feat_dict_trans[pos] = trans_l





# Write original sequences to fasta
with open(os.path.join(out_path, "original_sequences.fasta"), "w") as f:
    for pos in positions:
        for i, seq in enumerate(feat_dict_orig[pos]):
            f.write(f">pos_{pos}_seq_{i}\n{seq}\n")

# Write transition sequences to fasta 
with open(os.path.join(out_path, "transition_sequences.fasta"), "w") as f:
    for pos in positions:
        for i, seq in enumerate(feat_dict_trans[pos]):
            f.write(f">pos_{pos}_seq_{i}\n{seq}\n")


