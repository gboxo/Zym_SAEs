# %%
from transformers import AutoTokenizer, GPT2LMHeadModel
from config import update_cfg, post_init_cfg, get_default_cfg
import json
from config import get_default_cfg
import torch
from sae import BatchTopKSAE
from activation_store import ActivationsStore
from utils import get_ht_model, load_config, load_sae
from tqdm import tqdm



torch.set_grad_enabled(False)




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
            feature_activations = sparse_autoencoder(activation)["feature_acts"]
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
    feature_thresholds = torch.mean(all_feature_min_activations, dim=0)
    return feature_thresholds





if __name__ == "__main__":

    # Define the path of the SAE and load it
    path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_RAW_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_90000/"
    _,sae = load_sae(path)

    # Load the configuration file
    cfg = load_config("configs/workstation.yaml")
    cfg["ctx_len"] = 256
    cfg["model_batch_size"] = 64 


    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("AI4PD/ZymCTRL")
    model_ht = GPT2LMHeadModel.from_pretrained("AI4PD/ZymCTRL",
                                                    attn_implementation="eager",
                                                    torch_dtype=torch.float32)


    # Get the transformer lens model and compute the threshold
    config = model_ht.config
    config.d_mlp = 5120
    model = get_ht_model(model_ht,config, tokenizer=tokenizer)
    del model_ht
    # Compute the threshold and save it in the same directory as the SAE
    threshold = compute_threshold(model, sae, cfg, num_batches=100)
    torch.save(threshold, f"{path}/thresholds.pt")




