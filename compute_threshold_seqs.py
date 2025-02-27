# %%
from transformers import AutoTokenizer, GPT2LMHeadModel
from config import update_cfg, post_init_cfg, get_default_cfg
import json
from config import get_default_cfg
import torch
from sae import BatchTopKSAE
from activation_store import ActivationsStore
from utils import get_ht_model, load_config, load_sae, load_model
from tqdm import tqdm
import argparse


torch.set_grad_enabled(False)




def compute_threshold_from_sequences(model, sparse_autoencoder, num_seqs=100):
    """
    Computes a threshold for each feature based on the minimum positive activations across sequences.

    Args:
        model: The model from which activations are extracted.
        sparse_autoencoder: The sparse autoencoder used to compute feature activations.
        num_seqs: Number of sequences to process (default: 100).

    Returns:
        torch.Tensor: A tensor containing the computed threshold for each feature.
    """
    # Get activations in the same way as in debug.py
    test_set_path = "/users/nferruz/gboxo/Downloads/mini_brenda.txt"
    is_tokenized = False
    tokenizer = model.tokenizer
    
    with open(test_set_path, "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    test_set_tokenized = [tokenizer.encode(elem, padding=False, truncation=True, return_tensors="pt", max_length=256) for elem in test_set]

    names_filter = lambda x: x in "blocks.26.hook_resid_pre"
    
    all_feature_min_activations = []
    
    # Process sequences in batches to avoid GPU memory issues
    with torch.no_grad():
        for i, elem in enumerate(tqdm(test_set_tokenized[:num_seqs])):
            logits, cache = model.run_with_cache(elem.to("cuda"), names_filter=names_filter)
            activation = cache["blocks.26.hook_resid_pre"][0]
            
            # Process each activation with the SAE to get feature activations
            feature_activations = sparse_autoencoder.forward(activation)["feature_acts"]
            
            # For each feature, get the minimum activation that is greater than zero
            filtered_activations = torch.where(feature_activations > 0, feature_activations, float('inf'))
            feature_min_activations = torch.min(filtered_activations, dim=0).values
            feature_min_activations = torch.where(feature_min_activations == float('inf'), torch.nan, feature_min_activations)
            
            all_feature_min_activations.append(feature_min_activations)
            
            # Clean up to free GPU memory
            del feature_min_activations, filtered_activations, feature_activations, activation
            torch.cuda.empty_cache()

    all_feature_min_activations = torch.stack(all_feature_min_activations)
    all_feature_min_activations = torch.where(all_feature_min_activations > 0, all_feature_min_activations, 0)
    feature_thresholds = torch.nanmean(all_feature_min_activations, dim=0)
    
    # Handle any remaining NaN values
    feature_thresholds = torch.where(torch.isnan(feature_thresholds), torch.tensor(0.0, device=feature_thresholds.device), feature_thresholds)
    
    return feature_thresholds


def main(path, model_path, num_seqs):
    _,sae = load_sae(path)

    # Load the model and tokenizer
    tokenizer, model_ht = load_model(model_path)

    # Get the transformer lens model
    config = model_ht.config
    config.d_mlp = 5120
    model = get_ht_model(model_ht, config, tokenizer=tokenizer)
    
    # No need for activation store config anymore
    del model_ht
    
    # Compute the threshold using the sequences approach
    print(f"Computing thresholds from {num_seqs} sequences...")
    threshold = compute_threshold_from_sequences(model, sae, num_seqs=num_seqs)
    
    return threshold


if __name__ == "__main__":
    # Define the path of the SAE and load it
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae_path", type=str, required=False)
    parser.add_argument("--model_path", type=str, required=False)
    parser.add_argument("--num_seqs", type=int, required=False)
    args = parser.parse_args()

    if args.sae_path is None:
        path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000"
    else:
        path = args.sae_path

    if args.model_path is None:
        model_path = "AI4PD/ZymCTRL"
    else:
        model_path = args.model_path

    if args.num_seqs is None:
        num_seqs = 50000  # Default number of sequences to process
    else:
        num_seqs = args.num_seqs

    threshold = main(path, model_path=model_path, num_seqs=num_seqs)
    # Save with different name to distinguish from the original threshold computation method
    torch.save(threshold, f"{path}/thresholds_seqs.pt")
    print(f"Thresholds saved to {path}/thresholds_seqs.pt")



