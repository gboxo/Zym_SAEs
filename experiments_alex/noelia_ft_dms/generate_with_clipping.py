from argparse import ArgumentParser
from src.tools.generate.generate_utils import load_config
from sae_lens import HookedSAETransformer, SAE, SAEConfig
from src.utils import load_model, get_sl_model, load_sae
from functools import partial
import torch
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm

# Dictionary to store mask activation counts for each generation
mask_activations = {}

def clipping(activations:torch.Tensor, hook, clipping_feature:list[int], clipping_value:float = 5.0, generation_idx=None):

    # Check prompt processing
    if activations.shape[1] > 1:
        # Not a generation step, just processing the entire prompt
        # We don't need to record this
        pass
    else:
        # Where feature value is > 0, set it to clipping_value
        mask = activations[:,:,clipping_feature] > 0
        
        # Store mask information if we're in generation mode
        if generation_idx is not None:
            # Convert mask to CPU and detach from computation graph
            mask_cpu = mask.cpu().detach().numpy()
            
            # For each sample in the batch
            for i in range(mask_cpu.shape[0]):
                sample_key = f"sample_{i}"
                if sample_key not in mask_activations[generation_idx]:
                    mask_activations[generation_idx][sample_key] = []
                
                # Store whether mask was activated (1) or not (0) for this position
                # mask_cpu has shape [batch_size, 1, num_features]
                mask_activations[generation_idx][sample_key].append(mask_cpu[i, 0].tolist())
        
        # Apply the clipping
        activations[:,:,clipping_feature] = torch.where(mask, clipping_value, activations[:,:,clipping_feature])
    
    return activations


def generate_with_clipping(model: HookedSAETransformer, sae: SAE, prompt: str, clipping_feature: int, 
                          clipping_value: float = 5.0, max_new_tokens=256, n_samples=10, generation_idx=None):
    global mask_activations
    
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    
    all_outputs_batches = []
    batch_idx = 0

    for _ in tqdm(range(3)):  # Generate 20 batches
        new_generation_idx = generation_idx + "_" + str(batch_idx)
        
        # Initialize dictionary to store mask activations for this generation
        if new_generation_idx is not None:
            mask_activations[new_generation_idx] = {}
        
        clipping_hook = partial(
            clipping,
            clipping_feature=clipping_feature,
            clipping_value=clipping_value,
            generation_idx=new_generation_idx
        )
        
        # Use hooks for generation
        with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre.hook_sae_acts_post', clipping_hook)]):
            output = model.generate(
                input_ids_batch, 
                top_k=9,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
            )
        
        # Decode outputs
        all_outputs = model.tokenizer.batch_decode(output)
        all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]
        
        # Process the mask activation records to match output sequence lengths
        if new_generation_idx is not None:
            prompt_length = input_ids.shape[1]
            for i in range(n_samples):
                sample_key = f"sample_{i}"
                if sample_key in mask_activations[new_generation_idx]:
                    # Get actual sequence length (excluding padding)
                    seq_length = (output[i] != model.tokenizer.pad_token_id).sum().item()
                    # Only keep mask activations for actual generated tokens (excluding prompt)
                    generated_length = seq_length - prompt_length
                    if generated_length > 0:  # Ensure we generated at least one token
                        # Keep only the first 'generated_length' elements
                        mask_activations[new_generation_idx][sample_key] = mask_activations[new_generation_idx][sample_key][:generated_length]
        
        all_outputs_batches.append(all_outputs)
        batch_idx += 1

    return all_outputs_batches


cfg = SAEConfig(
    architecture="jumprelu",
    d_in=1280,
    d_sae=12*1280,
    activation_fn_str="relu",
    apply_b_dec_to_input=True,
    finetuning_scaling_factor=False,
    context_size=512,
    model_name="ZymCTRL",
    hook_name="blocks.25.hook_resid_pre",
    hook_layer=25,
    hook_head_index=None,
    prepend_bos=False,
    dtype="float32",
    normalize_activations="layer_norm",
    dataset_path=None,
    dataset_trust_remote_code=False,
    device="cuda",
    sae_lens_training_version=None,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)

    model_iteration = config["model_iteration"]
    data_iteration = config["data_iteration"]
    model_path = config["paths"]["model_path"]

    sae_path = config["paths"]["sae_path"]
    top_features_path = config["paths"]["top_features_path"]
    max_activations_path = config["paths"]["max_activations_path"]
    out_dir = config["paths"]["out_dir"]
    
    # Directory for mask activations data
    mask_dir = os.path.join(out_dir, "mask_activations")
    os.makedirs(mask_dir, exist_ok=True)

    cfg_sae, sae = load_sae(sae_path)
    thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
    thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
    state_dict = sae.state_dict()
    state_dict["threshold"] = thresholds
    del sae

    sae = SAE(cfg)
    sae.load_state_dict(state_dict)
    sae.use_error_term = True

    tokenizer, model = load_model(model_path)
    model = get_sl_model(model, model.config, tokenizer).to("cuda")
    model.add_sae(sae)

    with open(top_features_path, "rb") as f:
        important_features = pkl.load(f)
    x = important_features["coefs"][0]
    feature_indices = important_features["unique_coefs"]

    print("The features are:")

    print(feature_indices)



    with open(max_activations_path, "rb") as f:
        max_activations = pkl.load(f)
    
    # Create a dictionary to store the maximum activation for each feature (and one for all features)
    max_activation_dict = {f"feature_{i}": max_activations[i] for i in feature_indices}
    #max_activation_dict["feature_all"] = [max_activations[i] for i in range(feature_indices)]
        
    prompt = "3.2.1.1<sep><start>"
    os.makedirs(out_dir, exist_ok=True)

    # Dictionary to store all mask activation data (feature -> list of batch activations)
    all_mask_data = {}

    for i, clipping_feature in enumerate(tqdm(feature_indices)):
        # Use index as generation_idx to track mask activations for this feature
        max_activation = max_activation_dict[f"feature_{clipping_feature}"]
        generation_idx = f"feature_{clipping_feature}"
        out = generate_with_clipping(model, sae, prompt, clipping_feature, 
                                    max_new_tokens=1014, n_samples=20, 
                                    generation_idx=generation_idx,
                                    clipping_value=max_activation)
        
        # Save generated sequences
        with open(f"{out_dir}/clipping_feature_{clipping_feature}.txt", "w") as f:
            for batch_idx, batch_outputs in enumerate(out):
                for j, o in enumerate(batch_outputs):
                    f.write(f">3.2.1.1_batch{batch_idx}_{j},"+o+"\n")
        
        # Collect all batch activations for this feature
        feature_activations_batches = []
        for key in mask_activations.keys():
            if key.startswith(generation_idx):
                feature_activations_batches.append(mask_activations[key])
        all_mask_data[generation_idx] = feature_activations_batches
        with open(f"{mask_dir}/mask_activations_feature_{clipping_feature}.pkl", "wb") as f:
            pkl.dump(feature_activations_batches, f)
        
        # Optionally clear mask_activations entries here to save memory

    # Generate with all features
    if False:
        generation_idx = "feature_all"
        out = generate_with_clipping(model, sae, prompt, feature_indices, 
                                    max_new_tokens=1014, n_samples=20, 
                                    generation_idx=generation_idx)
        
        # Save generated sequences
        with open(f"{out_dir}/clipping_feature_all.txt", "w") as f:
            for batch_idx, batch_outputs in enumerate(out):
                for i, o in enumerate(batch_outputs):
                    f.write(f">3.2.1.1_batch{batch_idx}_{i},"+o+"\n")
        
        # Collect all batch activations for "all features"
        feature_activations_batches = []
        for key in mask_activations.keys():
            if key.startswith(generation_idx):
                feature_activations_batches.append(mask_activations[key])
        all_mask_data[generation_idx] = feature_activations_batches
        with open(f"{mask_dir}/mask_activations_feature_all.pkl", "wb") as f:
            pkl.dump(feature_activations_batches, f)
        
        # Save the complete mask activations dictionary with data from all features and all batches
        with open(f"{mask_dir}/all_mask_activations.pkl", "wb") as f:
            pkl.dump(all_mask_data, f)

    # Save the complete mask activations dictionary with data from all features and all batches
    with open(f"{mask_dir}/all_mask_activations.pkl", "wb") as f:
        pkl.dump(all_mask_data, f)

# Clean up
del model, sae 
torch.cuda.empty_cache()
