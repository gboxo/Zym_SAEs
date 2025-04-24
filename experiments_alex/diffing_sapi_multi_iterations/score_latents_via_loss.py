import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
from functools import partial
from torch.nn import CrossEntropyLoss
from tqdm import tqdm # Add tqdm for progress bars
import os # For path handling if needed
from sae_lens import SAEConfig, SAE

from sae_lens import HookedSAETransformer, SAE
from transformer_lens.hook_points import HookPoint
from src.utils import load_model, get_sl_model, load_sae
# Removed generate_with_ablation import if cfg is handled above
# from src.tools.generate.generate_with_ablation import generate_with_ablation, cfg


# Removed ablation_gen function

# --- Hook Function ---
def ablation_hook_fn(activations: torch.Tensor, hook:HookPoint, ablation_feature: list[int]):
    """Applies ablation by zeroing out specified features."""
    # Shape: (batch, sequence_pos, feature_dim) or (batch, feature_dim) during generation
    if activations.ndim == 3: # Batch, Sequence, Dim (likely during loss calculation)
        activations[:, :, ablation_feature] = 0
    elif activations.ndim == 2: # Batch, Dim (likely during generation token by token)
         activations[:, ablation_feature] = 0
    # Handle potential prompt processing shape (batch=1, seq=1, dim) if necessary
    elif activations.ndim == 3 and activations.shape[1] == 1:
         activations[:, :, ablation_feature] = 0
    else:
        print(f"Warning: Unexpected activation shape in hook: {activations.shape}")
    return activations

# Removed old ablation_loss function

# --- Helper Function for Per-Sequence Loss ---
def calculate_ce_loss_per_sequence(logits, tokens, attention_mask):
    """Calculates the mean cross-entropy loss for each sequence in a batch, ignoring padding."""
    # Shift so that tokens < n predict tokens n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction='none') # Get loss per token
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.shape) # Reshape to (batch, seq_len - 1)

    # Apply attention mask (shifted) to ignore padding tokens
    mask = attention_mask[..., 1:].contiguous()
    masked_loss = loss * mask

    # Calculate mean loss per sequence by dividing sum of loss by number of non-padding tokens
    # Add small epsilon to avoid division by zero for sequences with only one token (after shift)
    mean_loss_per_sequence = masked_loss.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    return mean_loss_per_sequence # Shape: (batch_size,)

# --- Updated Function to Calculate Losses (Baseline and Ablated) ---
def get_losses_batched(model, all_tokens, attention_mask, important_features, batch_size, hook_point):
    """Calculates baseline and ablated loss per sequence, processing in batches."""
    baseline_losses = []
    ablated_losses_dict = {} # Store ablated losses per feature index
    device = next(model.parameters()).device
    num_sequences = all_tokens.shape[0]

    # 1. Calculate Baseline Loss (No Hooks)
    print("Calculating baseline losses...")
    for j in range(0, num_sequences, batch_size):
        batch_start = j
        batch_end = min(j + batch_size, num_sequences)
        batch_tokens = all_tokens[batch_start:batch_end].to(device)
        batch_mask = attention_mask[batch_start:batch_end].to(device)
        with torch.no_grad():
            logits = model(batch_tokens, return_type="logits") # Get logits directly
            batch_losses = calculate_ce_loss_per_sequence(logits, batch_tokens, batch_mask)
            baseline_losses.extend(batch_losses.cpu().tolist())
    print(f"Baseline losses calculated for {len(baseline_losses)} sequences.")

    # 2. Calculate Ablated Losses
    num_ablation_conditions = len(important_features) + 1
    for i in tqdm(range(num_ablation_conditions), desc="Calculating Ablated Losses"):
        if i < len(important_features):
            # Ablate a single feature
            feature = [important_features[i]] # Hook expects a list
            index = str(i)
        else:
            # Ablate all important features together
            feature = important_features
            index = "all"

        # print(f"Calculating losses for ablation index: {index} (Feature(s): {feature})") # Optional print

        ablated_losses_dict[index] = [] # Initialize list for this condition
        # Define the hook function for this specific feature(s)
        ablation_hook = partial(ablation_hook_fn, ablation_feature=feature)

        # all_ablated_losses_for_feature = [] # Temp list for this feature ablation run (replaced by direct append)

        # Process data in batches
        for j in range(0, num_sequences, batch_size):
            batch_start = j
            batch_end = min(j + batch_size, num_sequences)
            batch_tokens = all_tokens[batch_start:batch_end].to(device)
            batch_mask = attention_mask[batch_start:batch_end].to(device)

            with torch.no_grad():
                with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre.hook_sae_acts_pre', ablation_hook)]):
                    logits = model(batch_tokens, return_type="logits")
                # Calculate per-sequence loss for the current batch
                batch_losses = calculate_ce_loss_per_sequence(logits, batch_tokens, batch_mask)
                # Extend the list with losses for sequences in this batch
                ablated_losses_dict[index].extend(batch_losses.cpu().tolist())

        # Store all calculated losses for this ablation condition
        # loss_dict[index] = all_ablated_losses_for_feature # Replaced by direct append
        # print(f"Completed index {index}, found {len(ablated_losses_dict[index])} loss values.") # Optional print


    return baseline_losses, ablated_losses_dict

# --- Function to Generate Sequences with Ablation ---
def generate_sequences_with_ablation(model, tokenizer, prompt: str, hook_point: str, ablation_feature: list[int], n_samples=10, max_new_tokens=100, **gen_kwargs):
    """Generates sequences with a given feature ablation."""
    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    # Repeat prompt for batch generation
    input_ids_batch = input_ids.repeat(n_samples, 1)
    print(input_ids_batch.shape)

    ablation_hook = partial(ablation_hook_fn, ablation_feature=ablation_feature)

    with torch.no_grad():
        with model.hooks(fwd_hooks=[('blocks.25.hook_resid_pre.hook_sae_acts_pre', ablation_hook)]):

            output_tokens = model.generate(
                input_ids_batch, 
                top_k=9, #tbd
                max_new_tokens=max_new_tokens,
                eos_token_id=1,
                do_sample=True,
                verbose=False,
                ) # Depending non your GPU, you'll be able to generate fewer or more sequences. This runs in an A40.
    


    # Decode generated sequences (excluding the prompt part)
    # output_tokens shape: [n_samples, seq_len]
    prompt_len = input_ids.shape[1]
    generated_parts = output_tokens[:, prompt_len:]
    all_outputs = tokenizer.batch_decode(generated_parts, skip_special_tokens=True)

    return all_outputs

# --- Function to Calculate Perplexity ---
def calculate_perplexity_batched(model, tokenizer, sequences: list[str], batch_size: int):
    """Calculates perplexity for a list of sequences using the base model."""
    device = next(model.parameters()).device
    perplexities = []
    num_sequences = len(sequences)

    # Determine the model's maximum sequence length (Prioritize model config)
    model_max_length = 1024
    # Prepend BOS token if necessary and available
    sequences_to_tokenize = []
    if tokenizer.bos_token:
        sequences_to_tokenize = [(tokenizer.bos_token + s) if not s.startswith(tokenizer.bos_token) else s for s in sequences]
    else:
        print("Warning: Tokenizer does not have a BOS token defined. Proceeding without prepending BOS.")
        sequences_to_tokenize = sequences


    print(f"Tokenizing {num_sequences} sequences for perplexity with max_length={model_max_length}...")
    tokenized_inputs = tokenizer(
        sequences_to_tokenize,
        return_tensors="pt",
        padding=True, # Pad to the longest sequence *in the batch* after truncation
        truncation=True, # Truncate sequences longer than max_length
        max_length=model_max_length # Use the determined max length
    )
    all_tokens = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]

    print(f"Tokenized inputs shape for perplexity calculation: {all_tokens.shape}") # Verify shape after tokenization

    print(f"Calculating perplexity for {num_sequences} sequences...")
    for j in range(0, num_sequences, batch_size):
        batch_start = j
        batch_end = min(j + batch_size, num_sequences)
        batch_tokens = all_tokens[batch_start:batch_end].to(device)
        batch_mask = attention_mask[batch_start:batch_end].to(device)

        # --- Debugging ---
        print(f"  Batch {j//batch_size + 1}: Feeding tokens with shape {batch_tokens.shape} to model.")
        if batch_tokens.shape[1] > model_max_length:
             print(f"  ERROR: Batch token length ({batch_tokens.shape[1]}) exceeds model max length ({model_max_length}) despite truncation!")
        # ---------------

        with torch.no_grad():
            # Pass attention_mask explicitly to the model call
            logits = model(batch_tokens, attention_mask=batch_mask, return_type="logits") # Error occurs here
            # Calculate loss per sequence for the current batch
            batch_losses = calculate_ce_loss_per_sequence(logits, batch_tokens, batch_mask)
            # Perplexity = exp(loss)
            batch_perplexities = torch.exp(batch_losses)
            perplexities.extend(batch_perplexities.cpu().tolist())

    return perplexities


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    model_path = "/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration6/"
    sae_path = "/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M1_D1/diffing/"
    top_features_path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/latent_scoring/latent_scoring_1_rl/important_features_topk/important_features_topk_pos_M1_D1.pkl"
    df_path = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/joined_dataframes/dataframe_all_iteration1.csv"
    # results_output_path = "ablation_effects_results.pkl" # Old path for aggregated results
    detailed_results_output_path = "ablation_detailed_results.pkl" # New path for detailed results
    # plot_output_path = "ablation_effects_plot.png" # Plotting is removed
    loss_batch_size = 32 # Batch size for loss calculation
    gen_batch_size = 10  # Number of samples to generate per ablation condition
    ppl_batch_size = 32 # Batch size for perplexity calculation
    max_gen_tokens = 1000 # Max new tokens for generation
    gen_top_k = 9      # Sampling param for generation

    # --- Load Data ---
    if False:
        print("Loading data...")
        df = pd.read_csv(df_path)
        original_sequences = df["sequence"].tolist()
        num_original_sequences = len(original_sequences) # Get number of original sequences

        with open(top_features_path, "rb") as f:
            important_features_data = pkl.load(f)
        # Adjust based on the actual structure of your .pkl file
        important_features = important_features_data[0]
        print(important_features)


        # --- Load Model, SAE, Tokenizer ---
        print("Loading SAE...")
        # Use the imported or defined sae_cfg here
        # Modify load_sae if it doesn't accept cfg or return state_dict separately
        try:
            # Attempt to load assuming load_sae returns cfg and state_dict
            cfg_sae_loaded, sae_state_dict = load_sae(sae_path, return_state_dict=True)
            # Use the loaded config if available, otherwise fall back to the defined one
            # cfg_for_sae = cfg_sae_loaded if cfg_sae_loaded else sae_cfg # sae_cfg defined later
        except TypeError:
            # Adapt if load_sae only returns the model
            print("Adapting SAE loading: Assuming load_sae returns the SAE model directly.")
            cfg_for_sae, sae_model_loaded = load_sae(sae_path) # Modify based on actual return
            sae_state_dict = sae_model_loaded.state_dict()
            del sae_model_loaded # Free memory
            # if not cfg_for_sae:
            #     cfg_for_sae = sae_cfg # Fallback # sae_cfg defined later

        sae_cfg = SAEConfig(
            d_in=1280,
            d_sae=12*1280,
            hook_name="blocks.25.hook_resid_pre", # Ensure this matches your hook point
            # Add other necessary parameters from your SAE config
            architecture="jumprelu",
            activation_fn_str="relu",
            apply_b_dec_to_input=True,
            finetuning_scaling_factor=False,
            context_size=512,
            model_name="ZymCTRL",
            hook_layer=25,
            hook_head_index=None,
            prepend_bos=False,
            dtype="float32",
            normalize_activations="layer_norm",
            device="cuda",
            dataset_path="",
            dataset_trust_remote_code=True,
            sae_lens_training_version=None

        )
        # Assign cfg_for_sae now that sae_cfg is defined

        sae = SAE(sae_cfg) # Use the determined config
        # Threshold loading - adapt path and key if necessary
        thresholds_path = os.path.join(sae_path, "percentiles/feature_percentile_50.pt") # More robust path joining
        try:
            thresholds = torch.load(thresholds_path)
            # Check if thresholds already exist in state_dict (e.g., from direct loading)
            if "thresholds" not in sae_state_dict and "threshold" not in sae_state_dict :
                sae_state_dict["threshold"] = thresholds # Use key expected by your SAE version
                print("Loaded thresholds.")
            elif "threshold" not in sae_state_dict:
                sae_state_dict["threshold"] = thresholds # Or maybe it's singular
                print("Loaded thresholds.")
            else:
                print("Thresholds seem to be already in state_dict.")
        except FileNotFoundError:
            print(f"Warning: Threshold file not found at {thresholds_path}. SAE might not use thresholds.")
        except Exception as e:
            print(f"Error loading thresholds: {e}")

        sae.load_state_dict(sae_state_dict)
        # sae.use_error_term = True # Configure based on your SAE

        print("Loading model and tokenizer...")
        tokenizer, model = load_model(model_path)
        model_config = model.config
        model_config.attn_implementation = "eager"
        # model_config.d_model = 5120 # d_model should be inferred, not hardcoded usually
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<pad>"

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = get_sl_model(model, model.config, tokenizer).to(device)
        model.add_sae(sae.to(device))

        hook_point = sae_cfg.hook_name # Get hook point from SAE config
        print(f"Hook point: {hook_point}")

        # --- Prepare Sequences for Loss Calculation ---
        prompt = "3.2.1.1<sep><start>"
        sequences_for_loss = [prompt + s for s in original_sequences]

        # Determine max length based on model config if available
        model_max_length = 1024

        tokenized_loss_inputs = tokenizer(
            sequences_for_loss,
            return_tensors="pt",
            padding="max_length", # Pad to model max length for consistent loss calculation
            truncation=True,
            max_length=model_max_length
        )
        loss_tokens = tokenized_loss_inputs["input_ids"]
        loss_mask = tokenized_loss_inputs["attention_mask"]

        # --- Calculate Baseline and Ablated Losses (Per Sequence) ---
        baseline_losses_per_sequence, ablated_losses_per_sequence_dict = get_losses_batched(
            model, loss_tokens, loss_mask, important_features, loss_batch_size, hook_point
        )
        # baseline_loss_avg = np.mean(baseline_losses_per_sequence) # Keep if needed for reference, but not primary output
        # print(f"Average Baseline Loss: {baseline_loss_avg:.4f}") # Optional print

        # --- Perform Ablation, Generation, and Perplexity Calculation (Detailed) ---
        # Initialize lists to store detailed generation results
        all_intervention_indices = []
        all_intervention_labels = []
        all_generated_sample_indices = []
        all_generated_sequences = []
        all_perplexities = []
        all_ablated_features = [] # Store which feature(s) were ablated for this gen

        num_ablation_conditions = len(important_features) + 1

        for i in tqdm(range(num_ablation_conditions), desc="Processing Ablation Conditions"):
            if i < len(important_features):
                feature = [important_features[i]]
                index = str(i)
                feature_label = f"Feature {important_features[i]}"
            else:
                feature = important_features
                index = "all"
                feature_label = "All Features"

            # 1. Generate Sequences with Ablation
            # print(f"\nGenerating sequences for {feature_label} (Index: {index})...") # Optional
            generated_seqs = generate_sequences_with_ablation(
                model, tokenizer, prompt, hook_point, feature,
                n_samples=gen_batch_size,
                max_new_tokens=max_gen_tokens,
                top_k=gen_top_k,
                # Add other generation params if needed
            )

            # 2. Calculate Perplexity of Generated Sequences (using base model)
            # print(f"Calculating perplexity for generated sequences (Index: {index})...") # Optional
            # Pass the model's max length to the perplexity function's tokenizer call
            generated_perplexities = calculate_perplexity_batched(
                model, tokenizer, generated_seqs, ppl_batch_size
            )

            # 3. Store detailed results for this intervention
            for sample_idx, (seq, ppl) in enumerate(zip(generated_seqs, generated_perplexities)):
                all_intervention_indices.append(index)
                all_intervention_labels.append(feature_label)
                all_generated_sample_indices.append(sample_idx)
                all_generated_sequences.append(seq)
                all_perplexities.append(ppl)
                all_ablated_features.append(feature) # Store the list of features used for this ablation

            # Optional: Print average perplexity for this condition during run
            # avg_perplexity = np.mean(generated_perplexities)
            # print(f"Index {index}: Avg Perplexity = {avg_perplexity:.4f}")


        # --- Structure and Save Detailed Results ---

        # 1. Create DataFrame for Loss Results
        loss_data = {"Sequence_Index": list(range(num_original_sequences)),
                    "Original_Sequence": original_sequences, # Add original sequence for reference
                    "Baseline_Loss": baseline_losses_per_sequence}
        for index, losses in ablated_losses_per_sequence_dict.items():
            loss_data[f"Ablated_Loss_{index}"] = losses
        loss_df = pd.DataFrame(loss_data)

        # 2. Create DataFrame for Generation/Perplexity Results
        gen_data = {"Intervention_Index": all_intervention_indices,
                    "Intervention_Label": all_intervention_labels,
                    "Ablated_Features": all_ablated_features,
                    "Generated_Sample_Index": all_generated_sample_indices,
                    "Generated_Sequence": all_generated_sequences,
                    "Perplexity": all_perplexities}
        gen_df = pd.DataFrame(gen_data)

        print("\nDetailed Loss Results DataFrame Head:")
        print(loss_df.head())
        print(f"\nLoss DataFrame Shape: {loss_df.shape}")

        print("\nDetailed Generation/Perplexity Results DataFrame Head:")
        print(gen_df.head())
        print(f"\nGeneration DataFrame Shape: {gen_df.shape}")


        # Save results (e.g., in a dictionary pickled)
        detailed_results = {
            "loss_results": loss_df,
            "generation_perplexity_results": gen_df
        }
        print(f"\nSaving detailed results dictionary to {detailed_results_output_path}")
        with open(detailed_results_output_path, "wb") as f:
            pkl.dump(detailed_results, f)

        # --- Remove Plotting Section ---
        # The previous plotting code relied on aggregated results_df which is no longer created.
        print("Plotting section removed as it requires aggregated data.")

        print("\nDone.")

# --- Configuration ---
# Load the detailed results

with open("ablation_detailed_results.pkl", "rb") as f:
    detailed_results = pkl.load(f)

loss_df = detailed_results["loss_results"]
gen_df = detailed_results["generation_perplexity_results"]



detailed_results_input_path = "ablation_detailed_results.pkl" # Path to the saved detailed results
plot_output_path = "ablation_effects_aggregated_plot.png" # Path to save the new plot

# --- Load Detailed Results ---
print(f"Loading detailed results from {detailed_results_input_path}...")
try:
    with open(detailed_results_input_path, "rb") as f:
        detailed_results = pkl.load(f)
    loss_df = detailed_results["loss_results"]
    gen_df = detailed_results["generation_perplexity_results"]
    print("Detailed results loaded successfully.")
except FileNotFoundError:
    print(f"Error: Results file not found at {detailed_results_input_path}")
    print("Please run the main script first to generate the detailed results.")
    exit()
except KeyError as e:
    print(f"Error: Missing key {e} in the loaded results dictionary.")
    print("Ensure the results file was generated by the correct script version.")
    exit()
except Exception as e:
    print(f"Error loading results file: {e}")
    exit()

# --- Aggregate Results ---
print("Aggregating detailed results...")
aggregated_results = []

# Get unique intervention indices/labels from the generation dataframe
intervention_groups = gen_df.groupby(["Intervention_Index", "Intervention_Label"])

for (index, label), group in intervention_groups:
    # 1. Calculate Average Perplexity for this intervention
    avg_perplexity = group['Perplexity'].mean()

    # 2. Calculate Average Loss Difference for this intervention
    if index == 'all':
        # Special case for 'all' features ablation
        ablated_loss_col = 'Ablated_Loss_all'
        feature_list = group['Ablated_Features'].iloc[0] # Get feature list for 'all'
    else:
        # For single feature ablations (index '0', '1', ...)
        ablated_loss_col = f'Ablated_Loss_{index}'
        # Get the single feature number
        try:
             feature_list = [int(label.split()[-1])] # Extract from "Feature XXX"
        except ValueError:
             feature_list = group['Ablated_Features'].iloc[0] # Fallback
             print(f"Warning: Could not parse feature number from label '{label}', using stored list.")


    if ablated_loss_col in loss_df.columns:
        # Calculate difference between ablated loss and baseline loss for each original sequence
        loss_diff_per_sequence = loss_df[ablated_loss_col] - loss_df['Baseline_Loss']
        # Average the difference across all original sequences
        avg_loss_diff = loss_diff_per_sequence.mean()
    else:
        print(f"Warning: Column '{ablated_loss_col}' not found in loss_df for index '{index}'. Setting loss diff to NaN.")
        avg_loss_diff = np.nan # Or handle as appropriate (e.g., 0 or skip)


    aggregated_results.append({
        "index": index,             # '0', '1', ..., 'all'
        "feature_label": label,     # 'Feature 123', 'All Features'
        "loss_diff_avg": avg_loss_diff,
        "perplexity_avg": avg_perplexity,
        "feature_list": feature_list # Store feature(s) for potential future use
    })

# Convert aggregated results to DataFrame
agg_results_df = pd.DataFrame(aggregated_results)

# Sort for potentially better plot labeling order (optional)
# Try sorting numerically then placing 'all' at the end
agg_results_df['sort_key'] = agg_results_df['index'].apply(lambda x: int(x) if x.isdigit() else float('inf'))
agg_results_df = agg_results_df.sort_values(by='sort_key').drop(columns='sort_key')


print("\nAggregated Results DataFrame:")
print(agg_results_df)

# --- Create Plot from Aggregated Data ---
print(f"\nGenerating plot from aggregated data and saving to {plot_output_path}...")
plt.figure(figsize=(12, 8))

# Create the scatter plot using the aggregated data
plot = sns.scatterplot(
    data=agg_results_df,
    x="loss_diff_avg",        # X-axis: Average increase in loss
    y="perplexity_avg",       # Y-axis: Average perplexity of generated sequences
    hue="index",              # Color points based on their index (optional)
    s=100,                    # Marker size
    legend=None               # Disable default legend, we'll add labels manually
)

# Add labels to each point, highlighting the "all" point
for i, row in agg_results_df.iterrows():
    label = row['index'] # Use the index ('0', '1', ..., 'all') as the label
    x_coord = row['loss_diff_avg']
    y_coord = row['perplexity_avg']

    # Skip labeling if coordinates are NaN (e.g., if loss column was missing)
    if pd.isna(x_coord) or pd.isna(y_coord):
        print(f"Skipping label for index '{label}' due to NaN coordinates.")
        continue

    # Styling for the "all" point
    if label == 'all':
        color = 'red'
        fontsize = 10
        fontweight = 'bold'
    else:
        color = 'black'
        fontsize = 8
        fontweight = 'normal'

    # Simple text offset to reduce overlap
    plt.text(x_coord + 0.01 * agg_results_df['loss_diff_avg'].std(), # Offset slightly horizontally
             y_coord + 0.01 * agg_results_df['perplexity_avg'].std(), # Offset slightly vertically
             label,
             color=color,
             fontsize=fontsize,
             fontweight=fontweight,
             ha='left', # Horizontal alignment
             va='bottom') # Vertical alignment

# Set plot title and labels
plt.title('Effect of SAE Feature Ablation on Loss Increase vs. Generated Perplexity (Aggregated)')
plt.xlabel('Average Increase in Loss (Ablated - Baseline)')
plt.ylabel('Average Perplexity of Generated Sequences')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.6)

# Adjust layout and save
plt.tight_layout()
plt.savefig(plot_output_path)
print(f"Plot saved to {plot_output_path}")

# Optionally display the plot directly
# plt.show()

print("\nDone.")
















