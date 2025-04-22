import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch import nn

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


#base_sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/finetune_SAE_DMS/diffing/checkpoint_latest.pt"
base_sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_25/sae_training_iter_0/final/checkpoint_latest.pt"


base_sae = torch.load(base_sae_path)
base_sae_dec = base_sae["model_state_dict"]["W_dec"]


cs_dict = [] 

for i in range(0, 5):
    bm_sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M0_D{i+1}/diffing/checkpoint_latest.pt"
    bm_sae = torch.load(bm_sae_path)
    bm_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M0_D{i+1}/diffing/thresholds.pt")
    bm_threshold = bm_threshold.cpu().numpy()
    # Inpute nan values with 0
    bm_threshold = np.nan_to_num(bm_threshold, nan=0)

    rl_sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M{i}_D{i}_rl/diffing/checkpoint_latest.pt"
    rl_sae = torch.load(rl_sae_path)
    rl_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M{i}_D{i}_rl/diffing/thresholds.pt")
    rl_threshold = rl_threshold.cpu().numpy()
    # Inpute nan values with 0
    rl_threshold = np.nan_to_num(rl_threshold, nan=0)
    

    bm_sae_dec = bm_sae["model_state_dict"]["W_dec"]
    rl_sae_dec = rl_sae["model_state_dict"]["W_dec"]

    x = cos(base_sae_dec, bm_sae_dec)
    y = cos(base_sae_dec, rl_sae_dec)

    # Store thresholds along with cosine similarities
    cs_dict.append((x, y, bm_threshold, rl_threshold))



# --- Plotting ---
# Create a figure and a set of subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10)) # Increased figsize slightly
axes = axes.flatten() # Flatten the 2D array of axes for easy iteration




# Define marker labels (used for legend)
marker_labels = {
    's': 'BM Threshold = 0',
    '^': 'RL Threshold = 0',
    'o': 'Both Thresholds != 0'
}
plotted_labels = set() # Keep track of labels already added to the legend

# Iterate through the collected data and plot each scatter plot
for i, (x_tensor, y_tensor, bm_thresh_tensor, rl_thresh_tensor) in enumerate(cs_dict):
    # Move tensors to CPU and convert to NumPy arrays for plotting
    x_np = x_tensor.cpu().numpy()
    y_np = y_tensor.cpu().numpy()

    ax = axes[i] # Select the subplot for the current iteration

    bm_thresh_np = bm_thresh_tensor
    rl_thresh_np = rl_thresh_tensor

    # --- Filter out points where both thresholds are zero ---
    mask_both_zero = (bm_thresh_np == 0) & (rl_thresh_np == 0)
    mask_to_plot = ~mask_both_zero

    x_plot = x_np[mask_to_plot]
    y_plot = y_np[mask_to_plot]
    bm_thresh_plot = bm_thresh_np[mask_to_plot]
    rl_thresh_plot = rl_thresh_np[mask_to_plot]

    # If no points remain after filtering, skip to the next subplot
    if x_plot.size == 0:
        ax.set_title(f"Iteration {i+1} (No valid data to plot)") # Updated title
        ax.set_xlabel(f"CS(Base, M0_D{i})")
        ax.set_ylabel(f"CS(Base, M{i}_D{i})")
        ax.grid(True)
        continue # Skip the rest of the plotting for this subplot

    # --- Determine Markers for plotted points ---
    markers = np.full(x_plot.shape, 'o', dtype=str) # Default to circle
    mask_sq = bm_thresh_plot == 0  # Square if BM is 0 (and RL is not, due to filtering)
    mask_tri = rl_thresh_plot == 0 # Triangle if RL is 0 (and BM is not, due to filtering)
    
    markers[mask_sq] = 's'
    markers[mask_tri] = '^'

    # --- Calculate Alpha for plotted points ---
    # Alpha inversely proportional to threshold difference, clipped between 0.1 and 1.0
    alpha_values = np.clip(np.abs(rl_thresh_plot - bm_thresh_plot), 0.1, 1.0)
    # --- Plot points based on marker type ---
    for marker_type in ['o', '^', 's']:
        mask = markers == marker_type
        if np.any(mask):
            label = marker_labels[marker_type]
            # Add label only if it hasn't been plotted before in this figure
            if label not in plotted_labels:
                ax.scatter(x_plot[mask], y_plot[mask], marker=marker_type, s=10, label=label, alpha=alpha_values[mask])
                plotted_labels.add(label)
            else:
                 ax.scatter(x_plot[mask], y_plot[mask], marker=marker_type, s=10, alpha=alpha_values[mask])


    ax.set_title(f"Iteration {i+1}") # Set subplot title
    ax.set_xlabel(f"CS(Base, M0_D{i})") # Set x-axis label
    ax.set_ylabel(f"CS(Base, M{i}_D{i})") # Set y-axis label
    ax.grid(True) # Add grid lines

# Hide any unused subplots (if the number of iterations is less than 6)
for j in range(len(cs_dict), len(axes)):
    fig.delaxes(axes[j])

# Add a single legend for the entire figure
fig.legend(loc='upper right', bbox_to_anchor=(0.99, 0.95))

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space for the figure legend

# Display the plot
plt.savefig("cs_plot_multi_iterations_markers_alpha_from_DMS_rl.png") # Changed filename




   
