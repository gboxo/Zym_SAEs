# %%
import os
from inference_batch_topk import convert_to_jumprelu
from utils import load_sae, load_model, get_ht_model
from sae import JumpReLUSAE
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import scipy.cluster.hierarchy as hierarchy
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

# %%
model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL"
tokenizer, model = load_model(model_path)
model = get_ht_model(model, model.config).to("cuda")
#sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg, sae = load_sae(sae_path)
thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
sae.to("cuda")
jump_relu = convert_to_jumprelu(sae, thresholds)
jump_relu.eval()
del sae

# %%

# We want to detect synthetic sequences for DNA Helicase (3.6.4.12)

def get_natural_and_synth_sequences():

    with open("/users/nferruz/gboxo/Downloads/mini_brenda.txt", "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    ec_numbers = [elem.split("<sep>")[0] for elem in test_set if len(elem.split("<sep>")) > 1]
    sequences = [elem.split("<sep>")[1] for elem in test_set if len(elem.split("<sep>")) > 1]
    ec_numbers = [elem for elem in ec_numbers if len(elem) > 0]
    indices = [i for i,elem in enumerate(ec_numbers) if elem == "3.6.4.12"]
    natural_sequences = [sequences[i].strip("<start>").strip("<end>").strip("<|endoftext|").strip("<end>") for i in indices]

    # %%

    files = os.listdir("DNA_helicase_generation")
    files = [file for file in files if file.endswith(".fasta")]
    synth_sequences = []
    for file in files:
        with open(f"DNA_helicase_generation/{file}", "r") as f:
            seq = f.read()
            seq = seq.split("\n")[1]
        synth_sequences.append(seq)

    # %%

    with open("nautral_sequenecs.txt", "w") as f:
        for seq in natural_sequences:
            f.write(seq + "\n")

    with open("synth_sequences.txt", "w") as f:
        for seq in synth_sequences:
            f.write(seq + "\n")

    return natural_sequences, synth_sequences


# %%

# ======= GET THE ACTIVAIONS ============

def get_activations( model, tokenizer, sequence):
    sequence = "3.6.4.12<sep>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    names_filter = lambda x: x.endswith("26.hook_resid_post")
    _, cache = model.run_with_cache(inputs, names_filter=names_filter)
    activations = cache["blocks.26.hook_resid_post"]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]
    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    return feature_acts


def get_all_features(model, sae, tokenizer, sequences):
    all_features = []
    for sequence in sequences:
        activations = get_activations(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features.detach().cpu().numpy())
    return all_features


# ======= GET THE FEATURES ============
with open("nautral_sequenecs.txt", "r") as f:
    natural_sequences = f.read()
    natural_sequences = natural_sequences.split("\n")
with open("synth_sequences.txt", "r") as f:
    synth_sequences = f.read()
    synth_sequences = synth_sequences.split("\n")

natural_features = get_all_features(model,jump_relu, tokenizer, natural_sequences)
synth_features = get_all_features(model,jump_relu, tokenizer, synth_sequences)

np.save("natural_features.npy",natural_features)
np.save("synth_features.npy",synth_features)



