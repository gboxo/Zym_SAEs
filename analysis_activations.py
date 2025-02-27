
import einops
import os
import torch
import numpy as np
from utils import load_sae, load_model, get_ht_model
from inference_batch_topk import convert_to_jumprelu
from compute_threshold import main
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
import argparse




if __name__ == "__main__":

    model_path = "AI4PD/ZymCTRL"
    test_set_path = "micro_brenda.txt"
    is_tokenized = False
    tokenizer, model = load_model(model_path)
    model_config = model.config
    model_config.d_mlp = 5120
    model = get_ht_model(model,model_config).to("cuda")
    with open(test_set_path, "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    test_set_tokenized = [tokenizer.encode(elem, padding=False, truncation=True, return_tensors="pt", max_length=256) for elem in test_set]

    names_filter = lambda x: x in "blocks.26.hook_resid_pre"
    activations = []
    max_len = 0
    with torch.no_grad():
        for i, elem in enumerate(test_set_tokenized[:100]):
            logits, cache = model.run_with_cache(elem.to("cuda"))
            acts = cache["blocks.26.hook_resid_pre"]
            if acts.shape[1] > max_len:
                max_len = acts.shape[1]
            activations.append(acts.cpu())

    # Pad activations to the same length
    for i, act in enumerate(activations):
        activations[i] = torch.cat([act, torch.zeros(act.shape[0], max_len - act.shape[1], act.shape[2])], dim=1)

    # Stack activations
    activations = torch.stack(activations)
    # Average norm over batch
    mean_norm = torch.norm(activations, dim=-1).mean(dim=0)

sns.lineplot(x=range(mean_norm.shape[1]), y=mean_norm[0].numpy())
plt.show()
















