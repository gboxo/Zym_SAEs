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

# Dictionary to store ablation activation counts for each generation
ablation_activations = {}

def ablation(activations:torch.Tensor, hook, ablation_feature:list[int], generation_idx=None):
    # Check prompt processing
    if activations.shape[1] > 1:
        # Not a generation step, just processing the entire prompt
        pass
    else:
        # Track which features would have been activated before ablation
        if generation_idx is not None:
            # Check which features would have been activated (value > 0)
            pre_ablation_mask = activations[:,:,ablation_feature] > 0
            # Convert mask to CPU and detach from computation graph
            mask_cpu = pre_ablation_mask.cpu().detach().numpy()
            
            # For each sample in the batch
            for i in range(mask_cpu.shape[0]):
                sample_key = f"sample_{i}"
                if sample_key not in ablation_activations[generation_idx]:
                    ablation_activations[generation_idx][sample_key] = []
                
                # If ablation_feature is a list of multiple features
                if isinstance(ablation_feature, list) and len(ablation_feature) > 1:
                    # Store activation status for each feature
                    ablation_activations[generation_idx][sample_key].append(mask_cpu[i, 0, :].tolist())
                else:
                    # For single feature (or single-element list), just store 0/1
                    ablation_activations[generation_idx][sample_key].append(mask_cpu[i, 0].tolist())
        
        # Now perform the ablation (set to zero)
        activations[:,:,ablation_feature] = 0
    
    return activations


def generate_with_ablation(model: HookedSAETransformer, sae: SAE, prompt: str, ablation_feature: int, 
                          max_new_tokens=256, n_samples=10, generation_idx=None):
    global ablation_activations
    
    # Initialize dictionary to store ablation activations for this generation
    
    input_ids = model.to_tokens(prompt, prepend_bos=sae.cfg.prepend_bos)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    

    all_outputs_batches = []
    batch_idx = 0

    for _ in tqdm(range(3)):

        new_generation_idx = generation_idx + "_" + str(batch_idx)
        ablation_hook = partial(
            ablation,
            ablation_feature=ablation_feature,
            generation_idx=new_generation_idx
        )

        if generation_idx is not None:
            ablation_activations[new_generation_idx] = {}
        batch_idx += 1


    
        # standard transformerlens syntax for a hook context for generation
        with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre.hook_sae_acts_post', ablation_hook)]):
            output = model.generate(
                input_ids_batch, 
                top_k=9, #tbd
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
            )
        
        all_outputs = model.tokenizer.batch_decode(output)
        all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]

        # Process the activation records to match output sequence lengths
        if generation_idx is not None:
            prompt_length = input_ids.shape[1]
            for i in range(n_samples):
                sample_key = f"sample_{i}"
                if sample_key in ablation_activations[new_generation_idx]:
                    # Get actual sequence length (excluding padding)
                    seq_length = (output[i] != model.tokenizer.pad_token_id).sum().item()
                    # Only keep activations for actual generated tokens (excluding prompt)
                    generated_length = seq_length - prompt_length
                    if generated_length > 0:  # Ensure we generated at least one token
                        # Keep only the first 'generated_length' elements
                        ablation_activations[new_generation_idx][sample_key] = ablation_activations[new_generation_idx][sample_key][:generated_length]
        all_outputs_batches.append(all_outputs)
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
    print("Hello, this is the start")
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()

    

    cfg_path = args.cfg_path
    config = load_config(cfg_path)

    model_iteration = config["model_iteration"]
    data_iteration = config["data_iteration"]
    model_path = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"

    sae_path = config["paths"]["sae_path"]
    top_features_path = config["paths"]["top_features_path"]
    out_dir = config["paths"]["out_dir"]

    
    
    # Directory for ablation activations data

    ablation_dir = os.path.join(out_dir, "ablation_activations")
    print(ablation_dir)
    os.makedirs(ablation_dir, exist_ok=True)

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

    print("All file for ablation")
    


    with open(top_features_path, "rb") as f:
        important_features = pkl.load(f)
    x = important_features["coefs"]
    feature_indices = important_features["unique_coefs"]
    print(feature_indices)
    if len(feature_indices) == 0:
        print("No features to ablate")
        exit()
        
    prompt = "3.2.1.1<sep><start>"
    os.makedirs(out_dir, exist_ok=True)

    # Dictionary to store all ablation data
    all_ablation_data = {}

    for i, ablation_feature in enumerate(tqdm(feature_indices)):
        generation_idx = f"feature_{ablation_feature}"
        out = generate_with_ablation(model, sae, prompt, 
                                ablation_feature, 
                                max_new_tokens=1014, n_samples=20, 
                                generation_idx=generation_idx)
        
        # Save generated sequences with batch indexing
        with open(f"{out_dir}/ablation_feature_{ablation_feature}.txt", "w") as f:
            for batch_idx, batch_outputs in enumerate(out):
                for j, o in enumerate(batch_outputs):
                    f.write(f">3.2.1.1_batch{batch_idx}_{j},"+o+"\n")
        
        # Save ablation activation data for this feature and add to all_ablation_data
        feature_activations_batches = []
        for key in ablation_activations.keys():
            if key.startswith(generation_idx):
                feature_activations_batches.append(ablation_activations[key])
        all_ablation_data[generation_idx] = feature_activations_batches
        with open(f"{ablation_dir}/ablation_activations_feature_{ablation_feature}.pkl", "wb") as f:
            pkl.dump(feature_activations_batches, f)
        
        

    # Generate with all features
    generation_idx = "feature_all"
    out = generate_with_ablation(model, sae, prompt, feature_indices, 
                            max_new_tokens=1014, n_samples=20, 
                            generation_idx=generation_idx)
    
    # Save generated sequences with batch indexing
    with open(f"{out_dir}/ablation_feature_all.txt", "w") as f:
        for batch_idx, batch_outputs in enumerate(out):
            for i, o in enumerate(batch_outputs):
                f.write(f">3.2.1.1_batch{batch_idx}_{i},"+o+"\n")
    
    # Save ablation activation data for all features
    feature_activations_batches = []
    for key in ablation_activations.keys():
        if key.startswith(generation_idx):
            feature_activations_batches.append(ablation_activations[key])
    all_ablation_data[generation_idx] = feature_activations_batches
    with open(f"{ablation_dir}/ablation_activations_feature_all.pkl", "wb") as f:
        pkl.dump(feature_activations_batches, f)
    
    # Save the complete ablation activations dictionary with data from all features
    with open(f"{ablation_dir}/all_ablation_activations.pkl", "wb") as f:
        pkl.dump(all_ablation_data, f)

# Clean up
del model, sae 
torch.cuda.empty_cache()
