import pandas as pd
import argparse
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model, load_config
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import pickle as pkl
from src.tools.diffing.latent_scoring_config import LatentScoringConfig

"""
This is slightly modified version of the latent scoring script from the diffing repo, given that we are using a df from the DMS dataset
"""




def get_activations( model, tokenizer, sequence, label = "3.2.1.1", hook_point = "blocks.25.hook_resid_pre"):
    sequence = f"{label}<sep><start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x == hook_point 
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache[hook_point]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]


    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def get_all_features(model, sae, tokenizer, sequences, label, hook_point="blocks.25.hook_resid_pre"):
    all_features = []
    for sequence in tqdm(sequences):
        activations = get_activations(model, tokenizer, sequence, label, hook_point=hook_point)
        features = get_features(sae, activations)
        all_features.append(features)
        del activations, features
        torch.cuda.empty_cache()
    return all_features

def obtain_features(sequences, mutant, output_dir, label, hook_point):
    """
    Obtain features from natural sequences
    """
    features = get_all_features(model,jump_relu, tokenizer, sequences, label, hook_point=hook_point)
    features_dict = dict(zip(mutant, features))
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    pkl.dump(features_dict, open(f"{output_dir}/features/features_{model_name}.pkl", "wb"))

    # Compute max activations
    key = list(features_dict.keys())[0]
    eg = np.array(features_dict[key].todense())

    max_activations = np.zeros(eg.shape[1])
    for key in features_dict.keys():
        max_activations = np.maximum(max_activations, np.array(features_dict[key].todense()).max(axis=0))

    pkl.dump(max_activations, open(f"{output_dir}/features/max_activations.pkl", "wb"))


    del features
    torch.cuda.empty_cache()

def load_features(path):
    """
    Load features from a file
    """
    assert path.endswith(".pkl"), "File must end with .pkl"
    features = pkl.load(open(path, "rb"))

    return features



def remove_nan_values(sequences, activity, mutant):
    """
    Remove nan values from sequences, activity and mutant
    """
    sequences = np.array(sequences)
    activity = np.array(activity)
    mutant = np.array(mutant)

    is_nan = np.isnan(activity)
    sequences = sequences[~is_nan]
    activity = activity[~is_nan]
    mutant = mutant[~is_nan]

    return sequences.tolist(), activity.tolist(), mutant.tolist()



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)


    df_path = config["paths"]["df_path"]
    output_dir = config["paths"]["out_dir"]
    model_path = config["paths"]["model_path"]
    sae_path = config["paths"]["sae_path"]

    seq_col_id = config["df"]["seq_col_id"]
    pred_col_id = config["df"]["pred_col_id"]
    col_id = config["df"]["col_id"]

    model_name = config["model_name"]
    ec_label = config["label"]
    DMS = config["is_DMS"]
    hook_point = config["hook_point"]

    prefix_tokens = config["prefix_tokens"]
    percentiles = config["percentiles"]
    min_rest_fraction = config["min_rest_fraction"]

    latent_scoring_config = LatentScoringConfig(
            hook_point=hook_point,
            model_name=model_name,
            is_DMS=DMS,
            label=ec_label,
            df_path=df_path,
            model_path=model_path,
            sae_path=sae_path,
            out_dir=output_dir,
            seq_col_id=seq_col_id,
            pred_col_id=pred_col_id,
            col_id=col_id,
            prefix_tokens=prefix_tokens,
            percentiles=percentiles,
            min_rest_fraction=min_rest_fraction
            )

    
    
    # Create the directories
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)


    df = pd.read_csv(df_path)

    sequences = df[seq_col_id].tolist()
    activity = df[pred_col_id].tolist()
    mutant = df[col_id].tolist()

    
    if DMS:
        sequences, activity, mutant = remove_nan_values(sequences, activity, mutant)
    


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

    obtain_features(sequences, mutant, output_dir, ec_label, hook_point=hook_point)

