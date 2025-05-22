import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
# KL Divergence
from torch.nn import functional as F
from src.utils import load_model, get_ht_model
from datasets import load_from_disk



base_model_path  = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
dpo_model_path = "/home/woody/b114cb/b114cb23/DPO_amylase_run_SAPI_FT_v2/output_iteration3/"


tokenizer,base_model = load_model(base_model_path)

model_config = base_model.config
model_config.attn_implementation = "eager"
model_config.d_model = 5120
base_model = get_ht_model(base_model, model_config, tokenizer)



tokenizer,dpo_model = load_model(dpo_model_path)
dpo_model_config = dpo_model.config
dpo_model_config.attn_implementation = "eager"
dpo_model_config.d_model = 5120
dpo_model = get_ht_model(dpo_model, dpo_model_config, tokenizer)




if True:
    dataset = load_from_disk("/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_3/eval/")

    out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_kl_divergence.pkl"
    out_path_base = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_base_log_probs.pkl"
    out_path_dpo = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_dpo_log_probs.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenized_sequences = dataset["input_ids"]

    

if False:
    dataset = load_from_disk("/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_0/eval/")

    out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M0_kl_divergence.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tokenized_sequences = dataset["input_ids"]
    out_path_base = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M0_base_log_probs.pkl"
    out_path_dpo = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M0_dpo_log_probs.pkl"


if False:
    dataset_path = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
    dataset = pd.read_csv(dataset_path)
    sequences = dataset["mutated_sequence"].values
    sequences = sequences[:100]
    out_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/dms_kl_divergence.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prompt ="3.2.1.1<sep><start>"
    tokenized_sequences = [tokenizer(prompt+sequence,  padding="max_length", truncation=True, max_length=512)["input_ids"] for sequence in sequences]








# --- KL Divergence Calculation ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
dpo_model.to(device)
base_model.eval()
dpo_model.eval()





# Get the KL divergence for each sequence

def logits_to_log_probs(logits):
    return torch.nn.functional.log_softmax(logits, dim=-1)

def log_probs_to_kl_divergence(log_probs_1, log_probs_2):
    probs_2 = torch.exp(log_probs_2) + 1e-10
    kl_div = torch.nn.functional.kl_div(log_probs_1, probs_2, reduction='none')
    kl_div = kl_div.sum(dim=-1)
    kl_div = kl_div.squeeze(0)
    return kl_div


all_kl_divergences = {}
all_base_log_probs = []
all_dpo_log_probs = []
with torch.no_grad():
    for i,tokenized_sequence in tqdm(enumerate(tokenized_sequences)):
        tokenized_sequence = torch.tensor(tokenized_sequence).to(device)

        logits_base = base_model(tokenized_sequence)
        logits_dpo = dpo_model(tokenized_sequence)

        base_log_probs = logits_to_log_probs(logits_base)
        dpo_log_probs = logits_to_log_probs(logits_dpo)

        kl_divergence = log_probs_to_kl_divergence(base_log_probs, dpo_log_probs)
        kl_divergence = kl_divergence[:-1].cpu().numpy()
        all_kl_divergences[i] = kl_divergence
        all_base_log_probs.append(base_log_probs.cpu().numpy())
        all_dpo_log_probs.append(dpo_log_probs.cpu().numpy())
        del logits_base, logits_dpo, base_log_probs, dpo_log_probs
        torch.cuda.empty_cache()








with open(out_path, "wb") as f:
    pickle.dump(all_kl_divergences, f)

with open(out_path_base, "wb") as f:
    pickle.dump(all_base_log_probs, f)

with open(out_path_dpo, "wb") as f:
    pickle.dump(all_dpo_log_probs, f)