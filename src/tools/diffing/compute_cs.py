import torch
from argparse import ArgumentParser
from diffing_utils import load_config
import torch.nn.functional as F
import os
import re
import seaborn as sns
import matplotlib.pyplot as plt

"""
Things to compute:
    - For each type, each iteration CS with M0_D0
    - Iteration-wise CS
"""



def compute_CS(W_dec, W_dec_ref):
    cs = F.cosine_similarity(W_dec, W_dec_ref, dim=1)
    return cs


def load_decoders(base_sae_path, diffing_path, device, start_index=0, end_index=30):
    sae_dict = {}

    string = "M0_D0"
    sae_dict[string] = torch.load(base_sae_path, map_location=device)["model_state_dict"]["W_dec"].detach().cpu()

    for data_index in range(start_index,end_index):
        string = f"M0_D{data_index}"
        sae_dict[string] = torch.load(diffing_path + string + "/diffing/checkpoint_latest.pt", map_location=device)["model_state_dict"]["W_dec"].detach().cpu()

    for index in range(1,30):
        string = f"M{index}_D{index}"
        sae_dict[string] = torch.load(diffing_path + string + "/diffing/checkpoint_latest.pt", map_location=device)["model_state_dict"]["W_dec"].detach().cpu()

    return sae_dict



def get_cs(sae_dict):
    all_cs = {} 
    # All vs M0_D0
    for key in sae_dict.keys():
        if key == "M0_D0": continue
        cs = compute_CS(sae_dict[key], sae_dict["M0_D0"])
        all_cs[key+"_vs_M0_D0"] = cs

    # Stage wise CS FT
    for i in range(1,30):
        cs = compute_CS(sae_dict[f"M0_D{i}"], sae_dict[f"M0_D{i-1}"])
        all_cs[f"M0_D{i}_vs_M0_D{i-1}"] = cs

    # Stage wise CS RL
    for i in range(1,30):
        cs = compute_CS(sae_dict[f"M{i}_D{i}"], sae_dict[f"M{i-1}_D{i-1}"])
        all_cs[f"M{i}_D{i}_vs_M{i-1}_D{i-1}"] = cs

def plot_cs(all_cs,output_dir):
    """
    We plot all the scatterplots of the type:
    cs(M0_DX,M0_D0) vs cs(MX_DX,M0_D0)
    """

    for i in range(1,30):
        cs_m0d0 = all_cs[f"M0_D{i}_vs_M0_D0"]
        cs_mxd0 = all_cs[f"M{i}_D{i}_vs_M0_D0"]
        sns.scatterplot(x=cs_m0d0, y=cs_mxd0, alpha=0.5 )
        plt.title(f"CS(M0_D{i},M0_D0) vs CS(M{i}_D{i},M0_D0)")
        plt.xlabel("CS(M0_D{i},M0_D0)")
        plt.ylabel("CS(M{i}_D{i},M0_D0)")
        plt.savefig(f"{output_dir}/M0_D{i}_vs_M{i}_D{i}.png")
        plt.close()

    







def main(config):
    base_path = config["paths"]["base_sae"]
    diffing_path = config["paths"]["diffing"]
    output_dir = config["paths"]["output_dir"]

    files = os.listdir(diffing_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    md_indices = [re.findall(r'\d+', file) for file in files]
    sae_dict = load_decoders(base_path, diffing_path, device)
    all_cs = get_cs(sae_dict)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(all_cs, f"{output_dir}/all_cs.pt")
    return all_cs

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--config_path", type=str, default="")
    args = argparser.parse_args()
    config_path = args.config_path

    config = load_config(config_path)
    all_cs = main(config)
    plot_cs(all_cs,config["paths"]["output_dir"])





