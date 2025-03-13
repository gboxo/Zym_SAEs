import os
import torch
from src.training.activation_store import ActivationsStore
from src.utils import get_ht_model, load_config, load_sae, load_model
from tqdm import tqdm
from src.utils import get_paths
from src.config.load_config import load_experiment_config, convert_to_sae_config




# %%
def compute_threshold(model, sparse_autoencoder, config, num_batches=12):
    """
    Computes a threshold for each feature based on the minimum positive activations across batches.

    Args:
        model: The model from which activations are extracted.
        sparse_autoencoder: The sparse autoencoder used to compute feature activations.
        config: Configuration object for the activations store.
        num_batches: Number of batches to process (default: 12).

    Returns:
        torch.Tensor: A tensor containing the computed threshold for each feature.
    """
    activations_store = ActivationsStore(model, config)

    all_feature_min_activations = []
    for _ in tqdm(range(num_batches)):
        tokens = activations_store.get_batch_tokens()
        batch_activations = activations_store.get_activations(tokens)

        batch_feature_min_activations = []
        for activation in batch_activations:
            feature_activations = sparse_autoencoder(activation)["feature_acts"].cpu()
            # For each feature, get the minimum activation that is greater than zero
            filtered_activations = torch.where(feature_activations > 0, feature_activations, float('inf'))
            feature_min_activations = torch.min(filtered_activations, dim=0).values
            feature_min_activations = torch.where(feature_min_activations == float('inf'), torch.nan, feature_min_activations)
            batch_feature_min_activations.append(feature_min_activations)

        batch_feature_min_activations = torch.stack(batch_feature_min_activations)
        filtered_activations = torch.where(batch_feature_min_activations > 0, batch_feature_min_activations, float('inf'))
        batch_min_activations = torch.min(filtered_activations, dim=0).values
        batch_min_activations = torch.where(batch_min_activations == float('inf'), torch.nan, batch_min_activations)
        all_feature_min_activations.append(batch_min_activations)

        # Clean up to free GPU memory
        del batch_min_activations, batch_feature_min_activations, filtered_activations, feature_activations, batch_activations
        torch.cuda.empty_cache()

    all_feature_min_activations = torch.stack(all_feature_min_activations)
    all_feature_min_activations = torch.where(all_feature_min_activations > 0, all_feature_min_activations, 0)
    # Compute the deciles of the activations
    feature_deciles = torch.quantile(all_feature_min_activations, torch.linspace(0, 1, 101), dim=0)
    os.makedirs(path+"/percentiles", exist_ok=True)
    for i in range(feature_deciles.shape[0]):
        
        torch.save(feature_deciles[i], f"{path}/percentiles/feature_percentile_{i}.pt")
    feature_thresholds = torch.mean(feature_deciles, dim=0)
    return feature_thresholds


def main(path, model_path, n_batches, cfg):
    _,sae = load_sae(path)
    sae.eval()

    # Load the configuration file
    cfg["ctx_len"] = 256
    cfg["model_batch_size"] = 64 


    # Load the model and tokenizer
    tokenizer, model_ht = load_model(model_path)


    # Get the transformer lens model and compute the threshold
    config = model_ht.config
    config.d_mlp = 5120
    model = get_ht_model(model_ht,config, tokenizer=tokenizer)
    model.eval()
    del model_ht
    # Compute the threshold and save it in the same directory as the SAE
    threshold = compute_threshold(model, sae, cfg, num_batches=n_batches)
    return threshold




if __name__ == "__main__":

    paths = get_paths()
    model_path = paths.model_path
    config = "configs/diffing_exp1/config_3_bm.yaml"
    config = load_experiment_config(config)
    sae_cfg = convert_to_sae_config(config)
    n_batches = 5
    path = "/users/nferruz/gboxo/Diffing Alpha Amylase/M0_D3/diffing/"


    threshold = main(path, model_path=model_path, n_batches = n_batches, cfg=sae_cfg)
    torch.save(threshold, f"{path}/thresholds.pt")



