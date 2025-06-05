#!/usr/bin/env python3
"""
Unified experiment runner for generation interventions
"""

import os
import pickle
from pathlib import Path
from argparse import ArgumentParser
from typing import Dict, List, Any, Tuple
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np

from src.tools.generate.generation_context import GenerationContext, generate_with_ablation, generate_with_clampping
from src.tools.generate.steering_hooks import dense_steering_hook
from src.utils import load_model, get_sl_model, load_sae, get_ht_model, load_config
from sae_lens import SAE, SAEConfig
from functools import partial


def setup_model_and_sae(model_path: str, sae_path: str) -> Tuple[object, SAE]:
    """Common setup for model and SAE across experiments"""
    cfg_sae, sae = load_sae(sae_path)
    thresholds = torch.load(sae_path + "/percentiles/feature_percentile_50.pt")
    thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
    state_dict = sae.state_dict()
    state_dict["threshold"] = thresholds
    del sae

    cfg = SAEConfig(
        architecture="jumprelu",
        d_in=cfg_sae["act_size"],
        d_sae=cfg_sae["dict_size"],
        activation_fn_str="relu",
        apply_b_dec_to_input=True,
        finetuning_scaling_factor=False,
        context_size=cfg_sae["seq_len"],
        model_name="ZymCTRL",
        hook_name=cfg_sae["hook_point"],
        hook_layer=cfg_sae["layer"],
        hook_head_index=None,
        prepend_bos=False,
        dtype="float32",
        normalize_activations="layer_norm",
        dataset_path=None,
        dataset_trust_remote_code=False,
        device="cuda",
        sae_lens_training_version=None,
    )

    sae = SAE(cfg)
    sae.load_state_dict(state_dict)
    sae.use_error_term = True

    tokenizer, model = load_model(model_path)
    model = get_sl_model(model, model.config, tokenizer).to("cuda")
    model.add_sae(sae)
    
    return model, sae


def load_feature_indices(features_path: str) -> List[int]:
    """Load feature indices from pickle file"""
    with open(features_path, "rb") as f:
        important_features = pickle.load(f)
    return important_features["unique_coefs"] # Hardcoded


def save_generation_results(outputs: List[List[str]], output_path: str, prefix: str = "3.2.1.1"):
    """Save generation results to file"""
    with open(output_path, "w") as f:
        for batch_idx, batch_outputs in enumerate(outputs):
            for j, output in enumerate(batch_outputs):
                f.write(f">{prefix}_batch{batch_idx}_{j},{output}\n")


def run_feature_intervention_experiment(config: Dict[str, Any], intervention_type: str):
    """Unified function for ablation and clampping experiments"""
    print(f"Running {intervention_type} experiment...")
    
    # Setup
    model, sae = setup_model_and_sae(config["paths"]["model_path"], config["paths"]["sae_path"])
    feature_indices = load_feature_indices(config["paths"]["top_features_path"])
    
    if len(feature_indices) == 0:
        print("No features to process")
        return

    prompt = f"{config['label']}<sep><start>"
    out_dir = config["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    context = GenerationContext()
    all_activation_data = {}

    # Load max activations for clampping if needed
    max_activation_dict = {}
    if intervention_type == "clampping" and "max_activations_path" in config["paths"]:
        with open(config["paths"]["max_activations_path"], "rb") as f:
            max_activations = pickle.load(f)
        max_activation_dict = {i: max_activations[i] for i in feature_indices}

    # Process individual features
    for feature_idx in tqdm(feature_indices):
        generation_idx = f"feature_{feature_idx}"
        
        if intervention_type == "ablation":
            outputs, batch_context = generate_with_ablation(
                model, sae, prompt, [feature_idx],
                max_new_tokens=1014, n_samples=20, 
                generation_idx=generation_idx, n_batches=3
            )
        else:  # clampping
            clampping_value = [max_activation_dict.get(feature_idx, 5.0)]  # Single value in list
            outputs, batch_context = generate_with_clampping(
                model, sae, prompt, [feature_idx], clampping_value,
                max_new_tokens=1014, n_samples=20,
                generation_idx=generation_idx, n_batches=3
            )
        
        # Merge contexts and save results
        context.activations.update(batch_context.activations)
        save_generation_results(outputs, f"{out_dir}/{intervention_type}_feature_{feature_idx}.txt", config["label"])
        
        # Collect activation data
        feature_activations_batches = []
        for key in context.activations.keys():
            if key.startswith(generation_idx):
                feature_activations_batches.append(context.activations[key])
        all_activation_data[generation_idx] = feature_activations_batches

    # Process all features together
    generation_idx = "feature_all"
    if intervention_type == "ablation":
        outputs, batch_context = generate_with_ablation(
            model, sae, prompt, feature_indices,
            max_new_tokens=1014, n_samples=20,
            generation_idx=generation_idx, n_batches=3
        )
    else:  # clampping - use list of values for each feature
        clampping_values = [max_activation_dict.get(feature_idx, 5.0) for feature_idx in feature_indices]
        outputs, batch_context = generate_with_clampping(
            model, sae, prompt, feature_indices, clampping_values,
            max_new_tokens=1014, n_samples=20,
            generation_idx=generation_idx, n_batches=3
        )
    
    context.activations.update(batch_context.activations)
    save_generation_results(outputs, f"{out_dir}/{intervention_type}_feature_all.txt", config["label"])

    # Save all activation data
    activation_dir = os.path.join(out_dir, f"{intervention_type}_activations")
    context.save_activations(activation_dir)
    
    with open(f"{activation_dir}/all_activations.pkl", "wb") as f:
        pickle.dump(all_activation_data, f)

    print(f"{intervention_type.capitalize()} experiment completed. Results saved to {out_dir}")


def run_ablation_experiment(config: Dict[str, Any]):
    """Run ablation experiment"""
    run_feature_intervention_experiment(config, "ablation")


def run_clampping_experiment(config: Dict[str, Any]):
    """Run clampping experiment"""
    run_feature_intervention_experiment(config, "clampping")


def run_steering_experiment(config: Dict[str, Any]):
    """Run steering experiment replicating just_steering.py"""
    print("Running steering experiment...")
    
    model_path = config["paths"]["model_path"]
    df_path = config["paths"]["df_path"]
    out_dir = config["paths"]["out_dir"]
    label = config["label"]
    
    # Load data and model
    df = pd.read_csv(df_path)
    tokenizer, model = load_model(model_path)
    model = get_ht_model(model, model.config, tokenizer).to("cuda")
    
    prompt = f"{label}<sep><start>"
    
    # Calculate steering vectors
    steering_vectors = calculate_steering_vectors(model, prompt, df)
    
    # Save steering vectors
    os.makedirs(out_dir, exist_ok=True)
    for layer, vector in steering_vectors.items():
        torch.save(vector, f"{out_dir}/steering_vector_layer_{layer}.pt")
    
    # Run generation with steering
    layers_to_steer = [5, 10, 25]
    for target_layer in tqdm(layers_to_steer, desc="Steering Layer"):
        current_steering_vector = steering_vectors[target_layer]
        for steering_strength in tqdm(np.arange(-1, 1.01, 0.5), desc=f"Layer {target_layer} Strength", leave=False):
            if steering_strength == 0:
                continue
                
            steering_hook = partial(
                dense_steering_hook,
                steering_vector=current_steering_vector,
                strength=float(steering_strength)
            )
            
            input_ids = model.to_tokens(prompt, prepend_bos=False)
            input_ids_batch = input_ids.repeat(20, 1)
            all_outputs = []
            
            for batch_idx in range(3):
                with model.hooks(fwd_hooks=[(f"blocks.{target_layer}.hook_resid_pre", steering_hook)]):
                    output = model.generate(
                        input_ids_batch,
                        top_k=9,
                        max_new_tokens=500,
                        eos_token_id=1,
                        do_sample=True,
                        verbose=False,
                    )
                
                batch_outputs = model.tokenizer.batch_decode(output)
                batch_outputs = [o.split("<|endoftext|>")[0] for o in batch_outputs]
                all_outputs.extend(batch_outputs)
            
            # Save results
            output_filename = f"{out_dir}/steering_layer_{target_layer}_strength_{steering_strength:.2f}.txt"
            with open(output_filename, "w") as f:
                for i, o in enumerate(all_outputs):
                    f.write(f">{config["label"]}_{i},{o}\n")

    print(f"Steering experiment completed. Results saved to {out_dir}")


def calculate_steering_vectors(model, prompt: str, df: pd.DataFrame, hook_name: str, layers: List[int] = [5, 10, 25], threshold: float = 3.0) -> Dict[int, torch.Tensor]:
    """Calculate steering vectors from activity data"""
    activity = df["prediction2"].values # Hardcoded
    sequences = df["sequence"].values # Hardcoded

    # Get top/bottom sequences
    sequences_top_10 = sequences[activity > threshold] 
    sequences_bottom_90 = sequences[activity < threshold] 
    random_sequences_bottom_90 = sequences_bottom_90[
        np.random.randint(0, len(sequences_bottom_90), size=len(sequences_top_10))
    ]
    
    tokenized_top_10 = [model.to_tokens(prompt + s) for s in sequences_top_10]
    tokenized_bottom_90 = [model.to_tokens(prompt + s) for s in random_sequences_bottom_90]
    
    names_filter = lambda x: hook_name in x
    all_pos = {}
    all_neg = {}
    
    with torch.no_grad():
        for tokens in tokenized_top_10:
            logits, cache_pos = model.run_with_cache(tokens, names_filter=names_filter)
            for layer in layers:
                acts_pos = cache_pos[f"blocks.{layer}.{hook_name}"][:, -1]
                if layer not in all_pos:
                    all_pos[layer] = acts_pos.mean(dim=0)
                else:
                    all_pos[layer] += acts_pos.mean(dim=0)
                    
        for tokens in tokenized_bottom_90:
            logits, cache_neg = model.run_with_cache(tokens, names_filter=names_filter)
            for layer in layers:
                acts_neg = cache_neg[f"blocks.{layer}.{hook_name}"][:, -1]
                if layer not in all_neg:
                    all_neg[layer] = acts_neg.mean(dim=0)
                else:
                    all_neg[layer] += acts_neg.mean(dim=0)
    
    # Normalize and compute differences
    for key in all_pos:
        all_pos[key] = all_pos[key] / len(tokenized_top_10)
        all_neg[key] = all_neg[key] / len(tokenized_bottom_90)
    
    steering_vectors = {key: all_pos[key] - all_neg[key] for key in all_pos}
    return steering_vectors


def main():
    parser = ArgumentParser()
    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--experiment_type", type=str, 
                       choices=["ablation", "clampping", "steering"], 
                       required=True)
    args = parser.parse_args()
    
    config = load_config(args.cfg_path)
    
    if args.experiment_type == "ablation":
        run_ablation_experiment(config)
    elif args.experiment_type == "clampping":
        run_clampping_experiment(config)
    elif args.experiment_type == "steering":
        run_steering_experiment(config)


if __name__ == "__main__":
    main() 