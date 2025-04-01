"""
Code for evaluating Sparse Autoencoders

We will evaluate the performance of the SAE in a hold out test set of 1000 sequences.


1) Load the SAE model
2) Compute the threshold if necessary
3) Covert BatchTopK to JumpReLU
3) Load the model tokenizer
4) Load the test set and tokenize it
5) Evaluate the SAE for the following metrics:
    - Reconstruction loss
    - Number of active features per sequence
    - Cosine Similarity 
"""
import pandas as pd
import os
import torch
import numpy as np
from src.utils import load_sae, load_model, get_ht_model
from src.inference.inference_batch_topk import convert_to_jumprelu
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable
from src.utils import get_paths
from datasets import load_from_disk
import seaborn as sns   
import matplotlib.pyplot as plt


class EvalConfig:
    model_path: str
    sae_path: str
    test_set_path: str
    is_tokenized: bool



class SAEEval:
    def __init__(self, 
                 cfg: EvalConfig):
        self.cfg = cfg
        torch.cuda.empty_cache()

        self.tokenizer, model = load_model(cfg.model_path)
        self.load_test_set()
        sae_cfg, sae = load_sae(cfg.sae_path, load_thresholds=False)
        thresholds = torch.load(cfg.sae_path+"/percentiles/feature_percentile_50.pt")
        self.sae_cfg = sae_cfg
        self.hook_point = sae_cfg["hook_point"]
        self.jump_relu = convert_to_jumprelu(sae, thresholds)
        self.jump_relu.to("cuda").eval()
        model_config = model.config
        model_config.d_mlp = 5120
        self.model = get_ht_model(model,model_config).to("cuda")
        del model




    def load_test_set(self):
        if not self.cfg.test_set_path.endswith(".txt"):
            dataset = load_from_disk(self.cfg.test_set_path)
            dataset = iter(dataset)
            self.test_set = [torch.tensor(next(dataset)["input_ids"]) for _ in range(100)]

            

        else:
            with open(self.cfg.test_set_path, "r") as f:
                test_set = f.read()
            test_set = test_set.split("\n")[:1000]
            test_set = [seq.strip("<pad>") for seq in test_set]
            test_set_tokenized = [self.tokenizer.encode(elem, return_tensors="pt", max_length=512) for elem in test_set]
            self.test_set = test_set_tokenized


    def evaluation_loop(self):
        all_activations = []
        all_sae_outputs = []
        
        for seq in tqdm(self.test_set):
            with torch.no_grad():
                names_filter = lambda x: self.hook_point in x
                _, cache = self.model.run_with_cache(seq, names_filter=names_filter)
                activations = cache[self.hook_point]
                sae_out = self.jump_relu.forward(activations, use_pre_enc_bias=True)
                out = {}
                for key, value in sae_out.items():
                    out[key] = value.cpu()
                del sae_out
                all_activations.append(activations.cpu())
                all_sae_outputs.append(out)
        
        # Compute metrics across all sequences at once
        self.final_metrics = self.get_metrics_across_sequences(all_activations, all_sae_outputs)

    def get_metrics_across_sequences(self, all_activations, all_sae_outputs):
        # Concatenate feature activations from all sequences
        all_feature_acts = torch.cat([out["feature_acts"][0] for out in all_sae_outputs], dim=0).cpu()

        has_fired = torch.where(all_feature_acts > 0, 1, 0)
        has_fired_per_token = torch.sum(has_fired, dim=1)
        has_fired_per_token = has_fired_per_token.cpu().numpy()
        os.makedirs(os.path.join(self.cfg.sae_path, "evaluation"), exist_ok=True)
        sns.histplot(has_fired_per_token)
        plt.xlabel("Number of features fired per token")
        plt.xlim(0, 300)
        plt.savefig(os.path.join(self.cfg.sae_path, "evaluation", "has_fired_per_token.png"))
        plt.close()
        # Calculate MSE loss across all sequences
        mse_loss = torch.mean(torch.tensor([sae_out["loss"] for sae_out in all_sae_outputs]))
        
        # Calculate binary activations and firing statistics
        seq_len, n_features = all_feature_acts.shape
        binary_acts = torch.where(all_feature_acts > 0, 1, 0)
        total_fires = torch.sum(binary_acts)
        total_fires_per_feat = torch.sum(binary_acts, dim=0)
        
        # Calculate firing rates for each feature across all sequences
        firing_rates = total_fires_per_feat / seq_len
        
        # Create activation density histogram
        non_zero_acts = all_feature_acts[all_feature_acts > 0]
        if len(non_zero_acts) > 0:
            non_zero_acts_cpu = non_zero_acts.detach().cpu().numpy()
            hist_values, bin_edges = np.histogram(
                non_zero_acts_cpu, 
                bins=20,
                range=(0, float(torch.max(non_zero_acts).item()))
            )
            hist_values = torch.tensor(hist_values)
            bin_edges = torch.tensor(bin_edges)
            activation_density = {
                "hist_values": hist_values,
                "bin_edges": bin_edges
            }
        else:
            activation_density = {
                "hist_values": torch.zeros(20),
                "bin_edges": torch.linspace(0, 1, 21)
            }
        
        # Calculate additional metrics
        sparsity = 1 - (torch.count_nonzero(all_feature_acts) / all_feature_acts.numel())
        max_activation = torch.max(all_feature_acts)
        mean_activation = torch.mean(all_feature_acts[all_feature_acts > 0]) if torch.any(all_feature_acts > 0) else torch.tensor(0.0)
        active_features = torch.count_nonzero(total_fires_per_feat)
        
        return {
            "mse_loss": mse_loss,
            "total_fires": total_fires,
            "total_fires_per_feat": total_fires_per_feat,
            "firing_rates": firing_rates,
            "activation_density": activation_density,
            "sparsity": sparsity,
            "max_activation": max_activation,
            "mean_activation": mean_activation,
            "active_features": active_features,
            "has_fired_per_token": has_fired_per_token
        }

    def pretty_table(self):
        metrics = self.final_metrics
        
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"  # Left align metric names
        table.align["Value"] = "r"   # Right align values
        
        # Add rows with formatted values
        table.add_row(["MSE Loss", f"{metrics['mse_loss'].item():.4f}"])
        table.add_row(["Total Fires", f"{metrics['total_fires'].item():.0f}"])
        table.add_row(["Sparsity (%)", f"{metrics['sparsity'].item()*100:.2f}"])
        table.add_row(["Max Activation", f"{metrics['max_activation'].item():.4f}"])
        table.add_row(["Mean Non-zero Activation", f"{metrics['mean_activation'].item():.4f}"])
        table.add_row(["Active Features", f"{metrics['active_features'].item():.0f}"])
        
        # Calculate feature usage statistics
        feat_usage = metrics['total_fires_per_feat'].cpu().numpy()
        dead_features = np.sum(feat_usage == 0)
        total_features = len(feat_usage)
        
        table.add_row(["Dead Features", f"{dead_features}/{total_features} ({dead_features/total_features*100:.1f}%)"])
        
        print(table)
        
    def save_feature_statistics(self, output_dir=None):
        """Save feature firing rates and activation density to files"""
        if output_dir is None:
            output_dir = os.path.join(self.cfg.sae_path, "evaluation")
            
        os.makedirs(output_dir, exist_ok=True)
        
        metrics = self.final_metrics
        
        # Save firing rates
        firing_rates = metrics["firing_rates"].cpu().numpy()
        np.save(os.path.join(output_dir, "feature_firing_rates.npy"), firing_rates)
        
        # Save activation density histogram
        activation_density = metrics["activation_density"]
        hist_values = activation_density["hist_values"].cpu().numpy()
        bin_edges = activation_density["bin_edges"].cpu().numpy()
        
        np.savez(
            os.path.join(output_dir, "activation_density.npz"),
            hist_values=hist_values,
            bin_edges=bin_edges
        )
        
        # Create and save firing rate distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(firing_rates, bins=50)
        plt.xlabel('Firing Rate')
        plt.ylabel('Number of Features')
        plt.title('Distribution of Feature Firing Rates')
        plt.savefig(os.path.join(output_dir, "firing_rate_distribution.png"))
        plt.close()
        
        # Create and save activation density plot
        plt.figure(figsize=(10, 6))
        plt.bar(
            bin_edges[:-1], 
            hist_values, 
            width=np.diff(bin_edges)[0],
            align='edge'
        )
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Non-zero Activation Values')
        plt.savefig(os.path.join(output_dir, "activation_density.png"))
        plt.close()
        
        print(f"Feature statistics saved to {output_dir}")



if __name__ == "__main__":
    #model_iteration = 1
    #data_iteration = 1
    #model_path = f"/users/nferruz/gboxo/Alpha Amylase/output_iteration{model_iteration}" 
    #sae_path = f"/users/nferruz/gboxo/Diffing Alpha Amylase/M{model_iteration}_D{data_iteration}/diffing/"
    #df_path = f"/users/nferruz/gboxo/Alpha Amylase/dataframe_iteration{data_iteration}.csv"
    #df = pd.read_csv(df_path)
    #sequences = df["sequence"].tolist()
    #txt = "\n".join(sequences)
    #test_set_path = f"/users/nferruz/gboxo/Alpha Amylase/test_set_iteration{data_iteration-1}.txt"
    #with open(test_set_path, "w") as f:
    #    f.write(txt)

    model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/New_SAE/sae_training_iter_0_32/final"
    test_set_path = "/home/woody/b114cb/b114cb23/boxo/new_dataset_eval/"
    #sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/New_SAE/sae_training_iter_0_32/final/"
    #test_set_path = "/home/woody/b114cb/b114cb23/boxo/new_dataset_concat_train/"






    if "woody" in test_set_path:
        config_path = "configs/base_config_alex.yaml"
    else:
        config_path = "configs/base_config_workstation.yaml"
    

    cfg = EvalConfig()
    cfg.model_path = model_path
    cfg.sae_path = sae_path
    cfg.test_set_path = test_set_path
    cfg.is_tokenized = False
    eval = SAEEval(cfg)
    eval.evaluation_loop()
    eval.pretty_table()
    eval.save_feature_statistics()  # Save feature statistics





