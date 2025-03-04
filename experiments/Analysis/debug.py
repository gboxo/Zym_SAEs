from inference_batch_topk import convert_to_jumprelu
from utils import load_sae, load_model, get_ht_model
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_acts():
    model_path = "AI4PD/ZymCTRL"
    test_set_path = "micro_brenda.txt"
    is_tokenized = False
    tokenizer, model = load_model(model_path)
    model_config = model.config
    model_config.d_mlp = 5120
    model = get_ht_model(model,model_config).to("cuda")
    with open(test_set_path, "r") as f:
        test_set = f.read()
    test_set = test_set.split("\n")
    test_set = [seq.strip("<pad>") for seq in test_set]
    test_set = [elem for seq in test_set for elem in seq.split("<|endoftext|>")]
    test_set_tokenized = [tokenizer.encode(elem, padding=False, truncation=True, return_tensors="pt", max_length=256) for elem in test_set]

    names_filter = lambda x: x in "blocks.26.hook_resid_pre"
    activations = []
    max_len = 0
    with torch.no_grad():
        for i, elem in enumerate(test_set_tokenized[:100]):
            logits, cache = model.run_with_cache(elem.to("cuda"))
            acts = cache["blocks.26.hook_resid_pre"]
            if acts.shape[1] > max_len:
                max_len = acts.shape[1]
            activations.append(acts.cpu())
    return activations, max_len





activations, max_len = get_acts()

sae_path = "/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/"
cfg, sae = load_sae(sae_path)

thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
print(thresholds.shape)
print(thresholds.min(), thresholds.max())


sae.to("cuda")

jump_relu = convert_to_jumprelu(sae, thresholds)

# Process multiple sequences instead of just the first one
num_seqs_to_process = min(1000, len(activations))  # Process up to 20 sequences
all_losses_sae = []
all_losses_jumprelu = []
all_binary_sae = []
all_binary_jumprelu = []

for i in range(num_seqs_to_process):
    # Process with BatchTopK SAE
    acts = activations[i].to("cuda")[0]
    x, x_mean, x_std = sae.preprocess_input(acts)
    sae_out = sae(acts)
    feature_acts_sae = sae_out["feature_acts"]
    reconstruct_post = sae_out["sae_out"]
    reconstruct_pre = (reconstruct_post - x_mean)/(x_std + 1e-6)
    l2 = (reconstruct_pre - x).pow(2).mean(dim=-1)
    all_losses_sae.append(l2.detach().cpu().numpy())
    all_binary_sae.append((feature_acts_sae > 0).cpu().numpy())
    
    # Process with JumpReLU
    x, x_mean, x_std = jump_relu.preprocess_input(acts)
    sae_out_jumprelu = jump_relu.forward(acts, use_pre_enc_bias=True)
    feature_acts = sae_out_jumprelu["feature_acts"]
    reconstruct_post = sae_out_jumprelu["sae_out"]
    reconstruct_pre = (reconstruct_post - x_mean)/(x_std + 1e-6)
    l2 = (reconstruct_pre - x).pow(2).mean(dim=-1)
    all_losses_jumprelu.append(l2.detach().cpu().numpy())
    all_binary_jumprelu.append((feature_acts > 0).cpu().numpy())

# Average losses for each position across sequences
avg_losses_sae = []
avg_losses_jumprelu = []
max_length = max(len(losses) for losses in all_losses_sae)

for pos in range(max_length):
    pos_losses_sae = [losses[pos] for losses in all_losses_sae if pos < len(losses)]
    pos_losses_jumprelu = [losses[pos] for losses in all_losses_jumprelu if pos < len(losses)]
    
    if pos_losses_sae:  # Only calculate average if there are values at this position
        avg_losses_sae.append(sum(pos_losses_sae) / len(pos_losses_sae))
    if pos_losses_jumprelu:
        avg_losses_jumprelu.append(sum(pos_losses_jumprelu) / len(pos_losses_jumprelu))

# Concatenate all binary activations
all_binary_sae_concat = np.concatenate(all_binary_sae)
all_binary_jumprelu_concat = np.concatenate(all_binary_jumprelu)

# Compute all the requested metrics

# 1. Percentage of dead features
n_features = all_binary_sae_concat.shape[1]  # Total number of features
active_features_sae = np.sum(np.sum(all_binary_sae_concat, axis=0) > 0)
active_features_jumprelu = np.sum(np.sum(all_binary_jumprelu_concat, axis=0) > 0)
dead_features_sae = n_features - active_features_sae
dead_features_jumprelu = n_features - active_features_jumprelu
dead_percentage_sae = (dead_features_sae / n_features) * 100
dead_percentage_jumprelu = (dead_features_jumprelu / n_features) * 100

# 2. Average number of fires per token
fires_per_token_sae = np.mean(np.sum(all_binary_sae_concat, axis=1))
fires_per_token_jumprelu = np.mean(np.sum(all_binary_jumprelu_concat, axis=1))

# 3. Variance of the loss
# Flatten all losses to compute overall variance
all_losses_sae_flat = np.concatenate([loss for loss in all_losses_sae])
all_losses_jumprelu_flat = np.concatenate([loss for loss in all_losses_jumprelu])
loss_variance_sae = np.var(all_losses_sae_flat)
loss_variance_jumprelu = np.var(all_losses_jumprelu_flat)

# Print metrics
print("\n===== Metrics Comparison =====")
print(f"{'Metric':<25} {'BatchTopK':<15} {'JumpReLU':<15}")
print(f"{'-'*25} {'-'*15} {'-'*15}")
print(f"{'Dead Features (%)':<25} {dead_percentage_sae:.2f}% {dead_percentage_jumprelu:.2f}%")
print(f"{'Active Features':<25} {active_features_sae}/{n_features} {active_features_jumprelu}/{n_features}")
print(f"{'Avg Fires Per Token':<25} {fires_per_token_sae:.2f} {fires_per_token_jumprelu:.2f}")
print(f"{'Loss Mean':<25} {np.mean(all_losses_sae_flat):.4f} {np.mean(all_losses_jumprelu_flat):.4f}")
print(f"{'Loss Variance':<25} {loss_variance_sae:.4f} {loss_variance_jumprelu:.4f}")

# Plot average losses
plt.figure(figsize=(10, 6))
sns.lineplot(x=range(len(avg_losses_jumprelu)), y=avg_losses_jumprelu, label="JumpReLU", color='b', linewidth=2, linestyle='-')
sns.lineplot(x=range(len(avg_losses_sae)), y=avg_losses_sae, label="BatchTopK", color='r', linewidth=2, linestyle='--')

plt.title("Average Loss Comparison Across Sequences", fontsize=16)
plt.xlabel("Position", fontsize=14)
plt.ylabel("Average Loss", fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Plot active features histogram combining all sequences
plt.figure(figsize=(10, 6))
sns.histplot(all_binary_sae_concat.sum(axis=1), color='r', label="BatchTopK", kde=True)
plt.axvline(fires_per_token_sae, color='r', linestyle='--', 
           label=f"BatchTopK Avg: {fires_per_token_sae:.2f}")

sns.histplot(all_binary_jumprelu_concat.sum(axis=1), color='b', label="JumpReLU", kde=True)
plt.axvline(fires_per_token_jumprelu, color='b', linestyle='--',
           label=f"JumpReLU Avg: {fires_per_token_jumprelu:.2f}")

plt.title("Active Features per Token (All Sequences)", fontsize=16)
plt.xlabel("Active Features", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.legend(fontsize=12)
plt.show()

# Plot feature usage to visualize dead features
plt.figure(figsize=(12, 6))
feature_usage_sae = np.sum(all_binary_sae_concat, axis=0)
feature_usage_jumprelu = np.sum(all_binary_jumprelu_concat, axis=0)

plt.subplot(1, 2, 1)
plt.bar(range(n_features), feature_usage_sae, width=1.0)
plt.title(f"BatchTopK Feature Usage\n({dead_percentage_sae:.1f}% Dead Features)")
plt.xlabel("Feature Index")
plt.ylabel("Number of Activations")

plt.subplot(1, 2, 2)
plt.bar(range(n_features), feature_usage_jumprelu, width=1.0)
plt.title(f"JumpReLU Feature Usage\n({dead_percentage_jumprelu:.1f}% Dead Features)")
plt.xlabel("Feature Index")
plt.ylabel("Number of Activations")

plt.tight_layout()
plt.show()






