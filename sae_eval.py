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
import einops
import os
import torch
import numpy as np
from utils import load_sae, load_model, get_ht_model
from inference_batch_topk import convert_to_jumprelu
from compute_threshold import main
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from prettytable import PrettyTable



class EvalConfig:
    model_path: str
    sae_path: str
    test_set_path: str
    is_tokenized: bool



class SAEEval:
    def __init__(self, cfg: EvalConfig):
        self.cfg = cfg
        torch.cuda.empty_cache()
        self.tokenizer, model = load_model(cfg.model_path)

        

        self.load_test_set()
        if "thresholds.pt" in os.listdir(cfg.sae_path):
            sae_cfg, sae, thresholds = load_sae(cfg.sae_path, load_thresholds=True)
        else:
            sae_cfg, sae = load_sae(cfg.sae_path, load_thresholds=False)
            thresholds = main(cfg.sae_path, cfg.model_path)
        self.sae_cfg = sae_cfg
        self.hook_point = sae_cfg["hook_point"]

        self.jump_relu = convert_to_jumprelu(sae, thresholds)

        self.jump_relu.to("cuda")
        model_config = model.config
        model_config.d_mlp = 5120
        self.model = get_ht_model(model,model_config).to("cuda")
        del model



    def load_test_set(self):
        if not self.cfg.is_tokenized:
            with open(self.cfg.test_set_path, "r") as f:
                test_set = f.read()
            test_set = test_set.split("\n")
            test_set = [seq.strip("<pad>") for seq in test_set]
            test_set_tokenized = [self.tokenizer.encode(elem, padding=True, truncation=True, return_tensors="pt", max_length=256) for elem in test_set]
            self.test_set = test_set_tokenized

        return self.test_set

    def evaluation_loop(self):
        metrics = []
        for seq in tqdm(self.test_set[:100]):
            with torch.no_grad():
                names_filter = lambda x: self.hook_point in x
                _, cache = self.model.run_with_cache(seq, names_filter=names_filter)
                activations = cache[self.hook_point]
                sae_out = self.jump_relu(activations)
                metrics.append(self.get_metrics(activations, sae_out))
        self.final_metrics = metrics




    def get_metrics(self, activations, sae_out):
        if not self.sae_cfg["input_unit_norm"]:
            mse_loss = torch.mean((activations - sae_out["sae_out"]) ** 2)
        else:
            pass
        feature_acts = sae_out["feature_acts"]
        binary_acts = torch.where(feature_acts > 0, 1, 0)
        total_fires = torch.sum(binary_acts)
        total_fires_per_feat = torch.sum(binary_acts, dim=0)
        
        # Add new metrics
        sparsity = 1 - (torch.count_nonzero(feature_acts) / feature_acts.numel())
        max_activation = torch.max(feature_acts)
        mean_activation = torch.mean(feature_acts[feature_acts > 0]) if torch.any(feature_acts > 0) else torch.tensor(0.0)
        active_features = torch.count_nonzero(total_fires_per_feat)

        return {
                "mse_loss": mse_loss,
                "total_fires": total_fires,
                "total_fires_per_feat": total_fires_per_feat,
                "sparsity": sparsity,
                "max_activation": max_activation,
                "mean_activation": mean_activation,
                "active_features": active_features
                }
    def process_metrics(self, metrics):
        all_mse_loss = []
        all_total_fires = []
        all_total_fires_per_feat = []
        all_sparsity = []
        all_max_activation = []
        all_mean_activation = []
        all_active_features = []
        
        for met in metrics:
            all_mse_loss.append(met["mse_loss"].cpu().numpy().item())
            all_total_fires.append(met["total_fires"])
            all_total_fires_per_feat.append(met["total_fires_per_feat"])
            all_sparsity.append(met["sparsity"].cpu().numpy().item())
            all_max_activation.append(met["max_activation"].cpu().numpy().item())
            all_mean_activation.append(met["mean_activation"].cpu().numpy().item())
            all_active_features.append(met["active_features"].cpu().numpy().item())

        all_total_fires_per_feat = torch.stack(all_total_fires_per_feat)
        all_total_fires_per_feat = einops.rearrange(all_total_fires_per_feat, "b s f -> (b s) f")
        all_total_fires_per_feat = all_total_fires_per_feat.cpu().numpy()
        all_total_fires = torch.stack(all_total_fires).cpu().numpy().mean()

        return {
            "mse_loss": np.mean(all_mse_loss),
            "total_fires": all_total_fires,
            "total_fires_per_feat": all_total_fires_per_feat,
            "sparsity": np.mean(all_sparsity),
            "max_activation": np.mean(all_max_activation),
            "mean_activation": np.mean(all_mean_activation),
            "active_features": np.mean(all_active_features)
        }



    def pretty_table(self):
        metrics = self.process_metrics(self.final_metrics)
        
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.align["Metric"] = "l"  # Left align metric names
        table.align["Value"] = "r"   # Right align values
        
        # Add rows with formatted values
        table.add_row(["MSE Loss", f"{metrics['mse_loss']:.4f}"])
        table.add_row(["Average Fires per Sequence", f"{metrics['total_fires']:.2f}"])
        table.add_row(["Sparsity (%)", f"{metrics['sparsity']*100:.2f}"])
        table.add_row(["Max Activation", f"{metrics['max_activation']:.4f}"])
        table.add_row(["Mean Non-zero Activation", f"{metrics['mean_activation']:.4f}"])
        table.add_row(["Active Features", f"{metrics['active_features']:.0f}"])
        
        # Calculate feature usage statistics
        feat_usage = metrics['total_fires_per_feat']
        dead_features = np.sum(feat_usage == 0)
        total_features = len(feat_usage)
        
        table.add_row(["Dead Features", f"{dead_features}/{total_features} ({dead_features/total_features*100:.1f}%)"])
        
        print("\nSparse Autoencoder Evaluation Metrics:")
        print(table)


if __name__ == "__main__":
    cfg = EvalConfig()
    cfg.model_path = "AI4PD/ZymCTRL"
    cfg.sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_RAW_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_90000"
    cfg.test_set_path = "/users/nferruz/gboxo/Downloads/micro_brenda.txt"
    cfg.is_tokenized = False

    eval = SAEEval(cfg)
    test_set = eval.load_test_set()
    eval.evaluation_loop()
    eval.pretty_table()





