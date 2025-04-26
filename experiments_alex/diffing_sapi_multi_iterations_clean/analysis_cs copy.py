import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch import nn
import pickle

cos = nn.CosineSimilarity(dim=1, eps=1e-6)


#base_sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/finetune_SAE_DMS/diffing/checkpoint_latest.pt"
base_sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_2025_04_02_32_15360_25/sae_training_iter_0/final/checkpoint_latest.pt"


def get_non_zero_indices(scoring):
    unique_coefs = scoring["unique_coefs"]
    activity_coeff = scoring["coefs"][0]
    non_zero_indices = torch.where(activity_coeff != 0)[0]
    non_zero_coefs = unique_coefs[non_zero_indices]
    return non_zero_coefs


scoring_dict = {}

for i in [1,2,3,4]:
    latent_scoring_path = f"/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/latent_scoring/"
    with open(f"{latent_scoring_path}/latent_scoring_{i}_bm/important_features_topk/important_features_topk_pos_M0_D{i}.pkl", "rb") as f:
        scoring_bm = pickle.load(f)
    #non_zero_coefs_bm = get_non_zero_indices(scoring_bm)
    non_zero_coefs_bm = scoring_bm[0]
    


    with open(f"{latent_scoring_path}/latent_scoring_{i}_rl/important_features_topk/important_features_topk_pos_M{i}_D{i}.pkl", "rb") as f:
        scoring_rl = pickle.load(f)
    #non_zero_coefs_rl = get_non_zero_indices(scoring_rl)
    non_zero_coefs_rl = scoring_rl[0]

    scoring_dict[i] = (non_zero_coefs_bm, non_zero_coefs_rl)

print(scoring_dict)








base_sae = torch.load(base_sae_path)
base_sae_dec = base_sae["model_state_dict"]["W_dec"]


cs_dict = [] 

for i in range(0, 5):
    bm_sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M0_D{i+1}/diffing/checkpoint_latest.pt"
    bm_sae = torch.load(bm_sae_path)
    bm_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M0_D{i+1}/diffing/thresholds.pt")
    bm_threshold = bm_threshold.cpu().numpy()
    # Inpute nan values with 0
    bm_threshold = np.nan_to_num(bm_threshold, nan=0)

    rl_sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M{i}_D{i}/diffing/checkpoint_latest.pt"
    rl_sae = torch.load(rl_sae_path)
    rl_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M{i}_D{i}/diffing/thresholds.pt")
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
    'o': 'Both Thresholds != 0',
    'green_circle': 'BM Important Feature', # New label for BM highlights
    'yellow_circle': 'RL Important Feature' # New label for RL highlights
}
plotted_labels_fig = set() # Keep track of labels already added to the FIGURE legend

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
        ax.set_xlabel(f"CS(Base, M0_D{i})") # Note: Indexing for labels might need review if M0_D{i+1} is intended
        ax.set_ylabel(f"CS(Base, M{i}_D{i})") # Note: Indexing for labels might need review if M{i+1}_D{i+1} is intended
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
    # Note: Consider if inverse proportionality is the desired effect. High difference = low alpha.
    # Maybe alpha proportional to the *sum* or *average* threshold? Or fixed alpha?
    # Using fixed alpha for now for simplicity, can be reverted.
    alpha_values = np.clip(np.abs(rl_thresh_plot - bm_thresh_plot), 0.1, 1.0)
    #alpha_values = np.full(x_plot.shape, 0.4) # Using a fixed alpha for clarity

    # --- Plot points based on marker type ---
    # Store handles and labels locally for this subplot's potential legend items
    local_handles, local_labels = [], []
    for marker_type in ['o', '^', 's']:
        mask = markers == marker_type
        if np.any(mask):
            label_key = marker_type
            label_text = marker_labels[label_key]
            # Add label only if it hasn't been plotted before IN THIS FIGURE
            current_label = label_text if label_text not in plotted_labels_fig else None
            scatter_plot = ax.scatter(x_plot[mask], y_plot[mask], marker=marker_type, s=10, label=current_label, alpha=alpha_values[mask])
            if current_label:
                 plotted_labels_fig.add(label_text)


    # --- Highlight Important Features ---
    iteration_key = i + 1 # scoring_dict keys are 1, 2, 3, 4
    if iteration_key in scoring_dict:
        # Ensure indices are on CPU and convert to numpy, then set for fast lookup
        bm_indices_tensor, rl_indices_tensor = scoring_dict[iteration_key]
        # Handle potential non-tensor data if loaded directly as numpy arrays
        if isinstance(bm_indices_tensor, torch.Tensor):
            bm_indices_set = set(bm_indices_tensor.cpu().numpy())
        else:
             bm_indices_set = set(bm_indices_tensor)
        if isinstance(rl_indices_tensor, torch.Tensor):
            rl_indices_set = set(rl_indices_tensor.cpu().numpy())
        else:
             rl_indices_set = set(rl_indices_tensor)


        original_indices_plotted = np.where(mask_to_plot)[0]

        is_bm_important_plotted = np.array([idx in bm_indices_set for idx in original_indices_plotted])
        is_rl_important_plotted = np.array([idx in rl_indices_set for idx in original_indices_plotted])

        # Plot green circles for BM important features
        if np.any(is_bm_important_plotted):
            label_key = 'green_circle'
            label_text = marker_labels[label_key]
            current_label = label_text if label_text not in plotted_labels_fig else None
            ax.scatter(x_plot[is_bm_important_plotted], y_plot[is_bm_important_plotted],
                       s=60, # Slightly larger size for circle visibility
                       facecolors='none',
                       edgecolors='green',
                       linewidths=1.5, # Make circle line thicker
                       label=current_label)
            if current_label:
                 plotted_labels_fig.add(label_text)


        # Plot yellow circles for RL important features
        if np.any(is_rl_important_plotted):
            label_key = 'yellow_circle'
            label_text = marker_labels[label_key]
            current_label = label_text if label_text not in plotted_labels_fig else None
            ax.scatter(x_plot[is_rl_important_plotted], y_plot[is_rl_important_plotted],
                       s=60, # Slightly larger size for circle visibility
                       facecolors='none',
                       edgecolors='yellow',
                       linewidths=1.5, # Make circle line thicker
                       label=current_label)
            if current_label:
                plotted_labels_fig.add(label_text)


    ax.set_title(f"Iteration {i+1}") # Set subplot title
    # Adjust axis labels based on the actual models being compared in iteration i
    # Assuming M0_D{i+1} vs M{i}_D{i}_rl based on file paths used earlier
    if i < 4 : # Check added to prevent index error for i=4 if rl_sae_path pattern holds
         ax.set_xlabel(f"CS(Base, M0_D{i+1})")
         ax.set_ylabel(f"CS(Base, M{i}_D{i})")
    else:
        # Handle label for the last iteration if needed, e.g., i=4
        ax.set_xlabel(f"CS(Base, M0_D{i+1})")
        ax.set_ylabel(f"CS(Base, ?)") # Placeholder if RL model naming pattern changes
    ax.grid(True) # Add grid lines

# Hide any unused subplots (if the number of iterations is less than 6)
for j in range(len(cs_dict), len(axes)):
    fig.delaxes(axes[j])

# Add a single legend for the entire figure
handles, labels = [], []
# Collect handles and labels from all axes
for ax in fig.axes:
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    for handle, label in zip(ax_handles, ax_labels):
        if label not in labels: # Ensure unique labels in the figure legend
            handles.append(handle)
            labels.append(label)

if handles: # Only add legend if there are items to show
    # Sort legend items (optional, but can make it tidier)
    # Example sort order: Markers first, then circles
    sort_order = {marker_labels['o']: 0, marker_labels['s']: 1, marker_labels['^']: 2,
                  marker_labels['green_circle']: 3, marker_labels['yellow_circle']: 4}
    handles_labels_sorted = sorted(zip(handles, labels), key=lambda x: sort_order.get(x[1], 99))
    handles, labels = zip(*handles_labels_sorted)
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95), title="Legend")


# Adjust layout to prevent overlapping titles/labels
plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust rect to make space for the figure legend

# Display the plot
plt.savefig("cs_plot_multi_iterations_markers_alpha_highlighted_from_DMS_clean_rl_topk.png") # Updated filename
plt.savefig("cs_plot_multi_iterations_markers_alpha_highlighted_from_DMS_clean_rl_topk.pdf", dpi=300) # Updated filename




   
