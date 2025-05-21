from argparse import ArgumentParser
from src.tools.generate.generate_utils import load_config
from transformer_lens import HookedTransformer
from src.utils import load_model, get_ht_model
from functools import partial
import torch
import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np

def ablation(activations:torch.Tensor, hook, steering_vector:torch.Tensor, strength:float):
    # Check prompt processing
    if activations.shape[1] == 1:
        activations = activations + steering_vector * strength
    
    return activations




def generate_with_steering(
    model: HookedTransformer, 
    prompt: str, 
    steering_vector_for_layer: torch.Tensor,
    target_layer: int,
    strength: float,
    max_new_tokens=256, 
    n_samples_per_batch=20, 
    n_batches=5
):
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    input_ids_batch = input_ids.repeat(n_samples_per_batch, 1)
    
    all_outputs_batches = []

    hook_tuple = (
        f"blocks.{target_layer}.hook_resid_pre",
        partial(
            ablation,
            steering_vector=steering_vector_for_layer,
            strength=strength,
        )
    )
    all_hooks = [hook_tuple]
    
    for batch_idx in tqdm(range(n_batches), desc=f"Layer {target_layer} Strength {strength:.2f} - Batch", leave=False):
        with model.hooks(fwd_hooks=all_hooks):
            output = model.generate(
                input_ids_batch, 
                top_k=9,
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
                )
        
        current_batch_outputs = model.tokenizer.batch_decode(output)
        current_batch_outputs = [o.split("<|endoftext|>")[0] for o in current_batch_outputs]
        current_batch_outputs = [o.replace("<|endoftext|>", "") for o in current_batch_outputs]
        all_outputs_batches.append(current_batch_outputs)

    return all_outputs_batches




    



def get_steering_vector(model: HookedTransformer, prompt: str, df:pd.DataFrame):
    # Get the empirical thresholds
    activity = df["prediction2"].values
    sequences = df["sequence"].values




    # Get top 10 sequences

    sequences_top_10 = sequences[activity > 3]
    tokenized_top_10 = [model.to_tokens(prompt + s) for s in sequences_top_10]
    # Get bottom 90 sequences
    sequences_bottom_90 = sequences[activity < 3]
    random_sequences_bottom_90 = sequences_bottom_90[np.random.randint(0, len(sequences_bottom_90), size=len(sequences_top_10))]
    tokenized_bottom_90 = [model.to_tokens(prompt + s) for s in random_sequences_bottom_90]


    names_filter = lambda x: "hook_resid_pre" in x

    all_pos = {}
    all_neg = {}

    with torch.no_grad():
        for i in range(len(tokenized_top_10)):
            logits,cache_pos = model.run_with_cache(tokenized_top_10[i], names_filter=names_filter)
            for layer in range(5,30,5):
                acts_pos = cache_pos[f"blocks.{layer}.hook_resid_pre"][:,-1]
                if layer not in all_pos.keys():
                    all_pos[layer] = acts_pos.mean(dim=0)
                else:
                    all_pos[layer] += acts_pos.mean(dim=0)
        for i in range(len(tokenized_bottom_90)):
            logits,cache_neg = model.run_with_cache(tokenized_bottom_90[i], names_filter=names_filter)
            for layer in range(5,30,5):
                acts_neg = cache_neg[f"blocks.{layer}.hook_resid_pre"][:,-1]
                if layer not in all_neg.keys():
                    all_neg[layer] = acts_neg.mean(dim=0)
                else:
                    all_neg[layer] += acts_neg.mean(dim=0)
        for key,val in all_pos.items():
            all_pos[key] = val / len(tokenized_top_10)
        for key,val in all_neg.items():
            all_neg[key] = val / len(tokenized_bottom_90)
            


    steering_vector = {}
    for key,val in all_pos.items():
        steering_vector[key] = all_pos[key] - all_neg[key]
    return steering_vector












if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)

    model_iteration = config["model_iteration"]
    data_iteration = config["data_iteration"]
    if model_iteration == 0 and data_iteration == 0:
        model_path = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
    else:
        model_path = config["paths"]["model_path"]

    sae_path = config["paths"]["sae_path"]
    df_path = config["paths"]["df_path"]
    out_dir = config["paths"]["out_dir"]

    df = pd.read_csv(df_path)




    tokenizer, model = load_model(model_path)
    model = get_ht_model(model, model.config, tokenizer).to("cuda")

    prompt = "3.2.1.1<sep><start>"
    steering_vector_dict = get_steering_vector(model, prompt, df)
    
    # Ensure out_dir exists before saving steering vectors
    os.makedirs(out_dir, exist_ok=True) 
    for key,val in steering_vector_dict.items():
        torch.save(val, f"{out_dir}/steering_vector_layer_{key}.pt")

    # Define number of samples per batch and number of batches
    N_SAMPLES_PER_BATCH = 20
    N_BATCHES = 3
    
    layers_to_steer = [5,10,25] # Layers [5, 10, 15, 20, 25]

    for target_layer in tqdm(layers_to_steer, desc="Steering Layer"):
        current_steering_vector = steering_vector_dict[target_layer]
        for steering_strength in tqdm(np.arange(-1, 1.01, 0.5), desc=f"Layer {target_layer} Strength", leave=False):
            if steering_strength == 0: # Skip strength 0 if it means no change
                continue

            out_batches = generate_with_steering(
                model, 
                prompt, 
                current_steering_vector,
                target_layer,
                strength=float(steering_strength), # Ensure strength is float
                max_new_tokens=500, 
                n_samples_per_batch=N_SAMPLES_PER_BATCH,
                n_batches=N_BATCHES
            )
            # Updated filename to include layer and strength
            output_filename = f"{out_dir}/steering_layer_{target_layer}_strength_{steering_strength:.2f}.txt"
            with open(output_filename, "w") as f:
                sample_idx_overall = 0
                for batch_outputs in out_batches:
                    for o in batch_outputs:
                        f.write(f">3.2.1.1_{sample_idx_overall},"+o+"\n")
                        sample_idx_overall += 1
    torch.cuda.empty_cache()
