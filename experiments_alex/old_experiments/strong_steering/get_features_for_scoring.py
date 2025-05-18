import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
from scipy.sparse import coo_matrix
from tqdm import tqdm
import pickle as pkl
"""
Here we just obtain the features for the sequences 
"""




def get_activations( model, tokenizer, sequence):
    sequence = "3.2.1.1<sep><start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("25.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.25.hook_resid_pre"]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]


    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def get_all_features(model, sae, tokenizer, sequences):
    all_features = []
    for sequence in tqdm(sequences):
        activations = get_activations(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features)
        del activations, features
        torch.cuda.empty_cache()
    return all_features

def obtain_features(sequences, mutant, output_dir):
    """
    Obtain features from natural sequences
    """
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    features_dict = dict(zip(mutant, features))
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    pkl.dump(features_dict, open(f"{output_dir}/features/features_M0_D0.pkl", "wb"))
    del features
    torch.cuda.empty_cache()

def read_sequence_from_file(path):
    """
    Reads **all** sequences in a .txt (multiple FASTA‐style entries).
    Each header line is of the form:
      >ID,… <sep> <start> A B C … <end>

    Returns:
      A dictionary mapping sequence IDs to sequence strings (with spaces removed)
    """
    seqs = {}
    curr = ""
    curr_id = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # If we were collecting a sequence, save it before starting a new one
                if curr and curr_id:
                    seqs[curr_id] = curr
                    curr = ""
                # Get the ID from the header line
                curr_id = line[1:].split(",")[0]
                # If this header has the entire seq on the same line:
                if "<start>" in line and "<end>" in line:
                    block = line.split("<start>", 1)[1].split("<end>", 1)[0]
                    curr = block.replace(" ", "")
                continue

            # Continue accumulating sequence tokens
            curr += line.replace(" ", "")

    # At EOF, flush last sequence
    if curr and curr_id:
        seqs[curr_id] = curr

    return seqs
 

if __name__ == "__main__":

    output_dir = "/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base/"
    seqs_path = "/home/woody/b114cb/b114cb23/boxo/strong_steering/seq_gen/seq_gen_3.2.1.1_ZC_FT.fasta"
    sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/strong_steering/diffing/"
    model_path = "/home/woody/b114cb/b114cb23/ZF_FT_alphaamylase_gerard/FT_3.2.1.1/"

    sequences = read_sequence_from_file(seqs_path)

    
    
    # Create the directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)

    
    



    if True:
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        # Load model
        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        torch.cuda.empty_cache()
        keys = list(sequences.keys())

        sequences = list(sequences.values())


        obtain_features(sequences, keys, output_dir)

