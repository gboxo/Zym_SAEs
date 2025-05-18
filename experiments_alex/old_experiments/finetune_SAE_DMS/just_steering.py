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

def ablation(activations:torch.Tensor, hook, steering_vector:torch.Tensor, strength:int):
    # Check prompt processing
    if activations.shape[1] == 1:
        activations = activations + steering_vector * strength
    
    return activations




def generate_with_steering(model: HookedTransformer, prompt: str, steering_vector: torch.Tensor, strength:int, max_new_tokens=256, n_samples=10):
    input_ids = model.to_tokens(prompt, prepend_bos=False)
    input_ids_batch = input_ids.repeat(n_samples, 1)
    

    all_outputs = []

    ablation_hook = partial(
        ablation,
        steering_vector=steering_vector,
        strength=strength,
    )
    
    #for i in range(n_samples):

    # standard transformerlens syntax for a hook context for generation
    with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre', ablation_hook)]):
        output = model.generate(
            input_ids_batch, 
            top_k=9, #tbd
            max_new_tokens=max_new_tokens,
            eos_token_id=1,
            do_sample=True,
            verbose=False,
            ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    
    all_outputs = model.tokenizer.batch_decode(output)
    all_outputs = [o.replace("<|endoftext|>", "") for o in all_outputs]


    return all_outputs

def get_empirical_thresholds(activity):
    """
    For each value compute the 0.25 and 0.75 percentile and use it as the lower and upper threshold
    """
    activity_quantiles = np.percentile(activity, [90, 90])
    thresholds_pos = {
        "pred": activity_quantiles[0],
    }
    thresholds_neg = {
        "pred": activity_quantiles[1],
    }
    return thresholds_pos, thresholds_neg



    



def get_steering_vector(model: HookedTransformer, prompt: str, df:pd.DataFrame):
    # Get the empirical thresholds
    activity = df["prediction"].values
    sequences = df["sequence"].values


    thresholds_pos, thresholds_neg = get_empirical_thresholds(activity)


    # Get top 10 sequences

    sequences_top_10 = sequences[activity > thresholds_pos["pred"]]
    tokenized_top_10 = [model.to_tokens(prompt + s) for s in sequences_top_10]
    # Get bottom 90 sequences
    sequences_bottom_90 = sequences[activity < thresholds_neg["pred"]]
    random_sequences_bottom_90 = sequences_bottom_90[np.random.randint(0, len(sequences_bottom_90), size=len(sequences_top_10))]
    tokenized_bottom_90 = [model.to_tokens(prompt + s) for s in random_sequences_bottom_90]


    names_filter = lambda x: "blocks.25.hook_resid_pre" in x

    all_pos = []
    all_neg = []

    with torch.no_grad():
        for i in range(len(tokenized_top_10)):
            logits,cache_pos = model.run_with_cache(tokenized_top_10[i], names_filter=names_filter)
            acts_pos = cache_pos["blocks.25.hook_resid_pre"][0]
            all_pos.append(acts_pos.mean(dim=0))
        for i in range(len(tokenized_bottom_90)):
            logits,cache_neg = model.run_with_cache(tokenized_bottom_90[i], names_filter=names_filter)
            acts_neg = cache_neg["blocks.25.hook_resid_pre"][0]
            all_neg.append(acts_neg.mean(dim=0))

    # Get the mean of the activations
    mean_pos = torch.stack(all_pos).mean(dim=0)
    print("Mean pos: ",mean_pos.shape)
    mean_neg = torch.stack(all_neg).mean(dim=0)
    steering_vector = mean_pos - mean_neg
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
        model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    else:
        model_path = config["paths"]["model_path"]

    sae_path = config["paths"]["sae_path"]
    df_path = config["paths"]["df_path"]
    out_dir = config["paths"]["out_dir"]

    df = pd.read_csv(df_path)




    tokenizer, model = load_model(model_path)
    model = get_ht_model(model, model.config, tokenizer).to("cuda")

    prompt = "3.2.1.1<sep><start>"
    os.makedirs(out_dir, exist_ok=True)
    steering_vector = get_steering_vector(model, prompt, df)
    torch.save(steering_vector, f"{out_dir}/steering_vector.pt")




    os.makedirs(out_dir, exist_ok=True)
    for steering_strength in range(-10,10,2):
        out = generate_with_steering(model, prompt, steering_vector, strength=steering_strength,max_new_tokens=1014, n_samples=20)
        with open(f"{out_dir}/steering_vector_{steering_strength}.txt", "w") as f:
            for i, o in enumerate(out):
                f.write(f">3.2.1.1_{i},"+o+"\n")
    torch.cuda.empty_cache()
