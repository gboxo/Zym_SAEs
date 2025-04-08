import torch
from argparse import ArgumentParser
from src.tools.diffing.diffing_utils import load_config
import torch.nn.functional as F
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures
from functools import partial
"""
Things to compute:
    - For each type, each iteration CS with M0_D0
    - Iteration-wise CS
"""



def compute_CS(W_dec, W_dec_ref):
    cs = F.cosine_similarity(W_dec, W_dec_ref, dim=1)
    return cs


def load_single_decoder(path_info, device):
    key, path = path_info
    try:
        decoder = torch.load(path, map_location=device)["model_state_dict"]["W_dec"]
        return key, decoder
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return key, None

def load_single_threshold(path_info, device):
    key, path = path_info
    try:
        threshold = torch.load(path, map_location=device)
        return key, threshold
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return key, None

def load_decoders(base_sae_path, diffing_path, device, start_index=0, end_index=30, max_workers=4):
    print(f"Loading decoders from {base_sae_path} and {diffing_path}")
    sae_dict = {}

    # Load base model
    sae_dict["M0_D0"] = torch.load(base_sae_path, map_location=device)["model_state_dict"]["W_dec"]

    # Prepare all paths
    paths_to_load = []
    # M0_DX paths
    paths_to_load.extend([(f"M0_D{i}", f"{diffing_path}M0_D{i}/diffing/checkpoint_latest.pt") 
                         for i in range(start_index, end_index)])
    # MX_DX paths
    paths_to_load.extend([(f"M{i}_D{i}", f"{diffing_path}M{i}_D{i}/diffing/checkpoint_latest.pt") 
                         for i in range(1, end_index)])

    # Load models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        load_fn = partial(load_single_decoder, device=device)
        futures = list(tqdm(
            executor.map(load_fn, paths_to_load),
            total=len(paths_to_load),
            desc="Loading models"
        ))
        
        # Add successfully loaded models to dictionary
        for key, decoder in futures:
            if decoder is not None:
                sae_dict[key] = decoder

    return sae_dict

def load_thresholds(base_sae_path, diffing_path, device, start_index=0, end_index=30, max_workers=4):
    print(f"Loading thresholds from {base_sae_path} and {diffing_path}")
    thresholds = {}

    # Load base model
    thresholds["M0_D0"] = torch.load(os.path.dirname(base_sae_path)+"/thresholds.pt", map_location=device)

    # Prepare all paths
    paths_to_load = []
    # M0_DX paths
    paths_to_load.extend([(f"M0_D{i}", f"{diffing_path}M0_D{i}/diffing/thresholds.pt") 
                         for i in range(start_index, end_index)])
    # MX_DX paths
    paths_to_load.extend([(f"M{i}_D{i}", f"{diffing_path}M{i}_D{i}/diffing/thresholds.pt") 
                         for i in range(1, end_index)])

    # Load models in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        load_fn = partial(load_single_threshold, device=device)
        futures = list(tqdm(
            executor.map(load_fn, paths_to_load),
            total=len(paths_to_load),
            desc="Loading thresholds"
        ))
        
        # Add successfully loaded models to dictionary
        for key, threshold in futures:
            if threshold is not None:
                thresholds[key] = threshold

    final_thresholds = {}
    for key,val in thresholds.items():
        val = torch.where(val > 0, val, torch.zeros_like(val))
        final_thresholds[key] = val.cpu().numpy()
    
    print(len(final_thresholds.keys()))
    return final_thresholds



def get_cs(sae_dict, start_index, end_index):
    print(f"Computing CS for {start_index} to {end_index}")
    all_cs = {} 
    # All vs M0_D0
    for key in tqdm(sae_dict.keys()):
        if key == "M0_D0": continue
        cs = compute_CS(sae_dict[key], sae_dict["M0_D0"])
        all_cs[key+"_vs_M0_D0"] = cs

    # Stage wise CS FT
    for i in tqdm(range(1, end_index)):
        cs = compute_CS(sae_dict[f"M0_D{i}"], sae_dict[f"M0_D{i-1}"])
        all_cs[f"M0_D{i}_vs_M0_D{i-1}"] = cs

    # Stage wise CS RL
    for i in tqdm(range(1, end_index)):
        cs = compute_CS(sae_dict[f"M{i}_D{i}"], sae_dict[f"M{i-1}_D{i-1}"])
        all_cs[f"M{i}_D{i}_vs_M{i-1}_D{i-1}"] = cs

    # Stage wise BM vs RL
    for i in tqdm(range(1, end_index)):
        cs = compute_CS(sae_dict[f"M{i}_D{i}"], sae_dict[f"M0_D{i}"])
        all_cs[f"M{i}_D{i}_vs_M0_D{i}"] = cs

    return all_cs

def plot_thresholds(output_dir,thresholds, end_index):
    """
    We plot all the scatterplots of the type:
    cs(M0_DX,M0_D0) vs cs(MX_DX,M0_D0)
    """
    print(f"Plotting CS for {end_index}")
    os.makedirs(f"{output_dir}/scatter_thresholds", exist_ok=True)
    for i in tqdm(range(1,end_index)):
        t_m0d0 = thresholds[f"M0_D{i}"]
        t_mxd0 = thresholds[f"M{i}_D{i}"]
        filter_features = (t_m0d0 > 0) | (t_mxd0 > 0)



        t_m0d0 = t_m0d0[filter_features]
        t_mxd0 = t_mxd0[filter_features]
        print(len(t_m0d0), len(t_mxd0))


        sns.scatterplot(x=t_m0d0, y=t_mxd0, alpha=0.5 )
        plt.title(f"Threshold(M0_D{i}) vs Threshold(M{i}_D{i})")
        plt.xlabel(f"Threshold(M0_D{i})")
        plt.ylabel(f"Threshold(M{i}_D{i})")
        plt.savefig(f"{output_dir}/scatter_thresholds/M0_D{i}_vs_M{i}_D{i}.png")
        plt.close()

def plot_decoder_norms(sae_dict,output_dir,thresholds, end_index):
    """
    We plot all the scatterplots of the type:
    cs(M0_DX,M0_D0) vs cs(MX_DX,M0_D0)
    """
    print(f"Plotting CS for {end_index}")
    os.makedirs(f"{output_dir}/scatter_decoder_norms", exist_ok=True)
    for i in tqdm(range(1,end_index)):
        t_m0d0 = thresholds[f"M0_D{i}"]
        t_mxd0 = thresholds[f"M{i}_D{i}"]
        filter_features = (t_m0d0 > 0) | (t_mxd0 > 0)

        m0d0_decoder = sae_dict[f"M0_D{i}"]
        mxd0_decoder = sae_dict[f"M{i}_D{i}"]




        # Compute the norms of the decoders
        norm_m0d0 = torch.norm(m0d0_decoder, dim=1)
        norm_mxd0 = torch.norm(mxd0_decoder, dim=1)
        norm_m0d0 = norm_m0d0[filter_features]
        norm_mxd0 = norm_mxd0[filter_features]
        norm_m0d0 = norm_m0d0.cpu().numpy().flatten()
        norm_mxd0 = norm_mxd0.cpu().numpy().flatten()






        sns.scatterplot(x=norm_m0d0, y=norm_mxd0, alpha=0.5 )
        plt.title(f"Decoder Norm(M0_D{i}) vs Decoder Norm(M{i}_D{i})")
        plt.xlabel(f"Decoder Norm(M0_D{i})")
        plt.ylabel(f"Decoder Norm(M{i}_D{i})")
        plt.savefig(f"{output_dir}/scatter_decoder_norms/M0_D{i}_vs_M{i}_D{i}.png")
        plt.close()

def plot_cs(all_cs,output_dir,thresholds, end_index):
    """
    We plot all the scatterplots of the type:
    cs(M0_DX,M0_D0) vs cs(MX_DX,M0_D0)
    """
    print(f"Plotting CS for {end_index}")
    os.makedirs(f"{output_dir}/scatter_cs", exist_ok=True)
    for i in tqdm(range(1,end_index)):
        t_m0d0 = thresholds[f"M0_D{i}"]
        t_mxd0 = thresholds[f"M{i}_D{i}"]
        filter_features = (t_m0d0 > 0) | (t_mxd0 > 0)
        print(filter_features.sum())



        cs_m0d0 = all_cs[f"M0_D{i}_vs_M0_D0"]
        cs_mxd0 = all_cs[f"M{i}_D{i}_vs_M0_D0"]
        cs_m0d0 = cs_m0d0[filter_features]
        cs_mxd0 = cs_mxd0[filter_features]
        print(len(cs_m0d0), len(cs_mxd0))


        sns.scatterplot(x=cs_m0d0, y=cs_mxd0, alpha=0.5 )
        plt.title(f"CS(M0_D{i},M0_D0) vs CS(M{i}_D{i},M0_D0)")
        plt.xlabel(f"CS(M0_D{i},M0_D0)")
        plt.ylabel(f"CS(M{i}_D{i},M0_D0)")
        plt.savefig(f"{output_dir}/scatter_cs/M0_D{i}_vs_M{i}_D{i}.png")
        plt.close()

def plot_cs_iteration_wise(all_cs, output_dir, thresholds, end_index):
    """
    We plot violin plots for each iteration showing the distribution of cosine similarities
    Split into two subplots: one for M0_D comparisons and one for M_D comparisons
    """
    print(f"Plotting CS for {end_index}")
    
    # Separate data for each type
    m0d_data = []
    m0d_labels = []
    md_data = []
    md_labels = []

    for i in tqdm(range(1, end_index)):
        # Convert tensors to numpy arrays
        t_m0d0 = thresholds[f"M0_D{i}"]
        t_mxd0 = thresholds[f"M{i}_D{i}"]
        filter_features = (t_m0d0 > 0) & (t_mxd0 > 0)
        
        cs_m0d0 = all_cs[f"M0_D{i}_vs_M0_D{i-1}"].cpu().numpy().flatten()
        cs_mxd0 = all_cs[f"M{i}_D{i}_vs_M{i-1}_D{i-1}"].cpu().numpy().flatten()
        cs_m0d0 = cs_m0d0[filter_features]
        cs_mxd0 = cs_mxd0[filter_features]
        
        # Store data separately
        m0d_data.extend(cs_m0d0)
        m0d_labels.extend([f"D{i} vs D{i-1}"] * len(cs_m0d0))
        
        md_data.extend(cs_mxd0)
        md_labels.extend([f"M{i}D{i} vs M{i-1}D{i-1}"] * len(cs_mxd0))

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot M0_D comparisons
    sns.violinplot(x=m0d_labels, y=m0d_data, ax=ax1)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    ax1.set_title("Distribution of Cosine Similarities (Fine-tuning)")
    ax1.set_xlabel("Model Comparison")
    ax1.set_ylabel("Cosine Similarity")
    
    # Plot M_D comparisons
    sns.violinplot(x=md_labels, y=md_data, ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_title("Distribution of Cosine Similarities (RL Training)")
    ax2.set_xlabel("Model Comparison")
    ax2.set_ylabel("Cosine Similarity")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/iteration_wise_cs_split.png")
    plt.close()


def main(config, start_index, end_index, max_workers=4):


    return all_cs







if __name__ == "__main__":
    start_index = 0
    end_index = 30
    argparser = ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="")
    args = argparser.parse_args()
    config_path = args.config_path

    config = load_config(config_path)
    base_path = config["paths"]["base_sae"]
    diffing_path = config["paths"]["diffing"]
    output_dir = config["paths"]["output_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if True:
        sae_dict = load_decoders(base_path, diffing_path, device, 
                                start_index=start_index, 
                                end_index=end_index,
                                max_workers=50)
        all_cs = get_cs(sae_dict, start_index=start_index, end_index=end_index)
        os.makedirs(output_dir, exist_ok=True)
        torch.save(all_cs, f"{output_dir}/all_cs.pt")
    else:
        all_cs = torch.load(f"{output_dir}/all_cs.pt")



    thresholds = load_thresholds(config["paths"]["base_sae"], config["paths"]["diffing"], "cpu", start_index=start_index, end_index=end_index, max_workers=50)
    #plot_cs(all_cs,config["paths"]["output_dir"],thresholds, end_index=end_index)
    #plot_thresholds(config["paths"]["output_dir"],thresholds, end_index=end_index)
    #plot_cs_iteration_wise(all_cs,config["paths"]["output_dir"],thresholds, end_index=end_index)
    plot_decoder_norms(all_cs,config["paths"]["output_dir"],thresholds, end_index=end_index)
    






