import torch
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

# KL Divergence
from torch.nn import functional as F
from src.utils import load_model, get_ht_model




base_model_path  = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
dpo_model_path = "/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration29/"

tokenizer,base_model = load_model(base_model_path)

model_config = base_model.config
model_config.attn_implementation = "eager"
model_config.d_model = 5120
base_model = get_ht_model(base_model, model_config)



tokenizer,dpo_model = load_model(dpo_model_path)
dpo_model_config = dpo_model.config
dpo_model_config.attn_implementation = "eager"
dpo_model_config.d_model = 5120
dpo_model = get_ht_model(dpo_model, dpo_model_config)


dataset_path = "/home/woody/b114cb/b114cb23/boxo/new_dataset_concat_train/new_dataset_concat_train"


dataset = load_from_disk(dataset_path)












# --- KL Divergence Calculation ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model.to(device)
dpo_model.to(device)
base_model.eval()
dpo_model.eval()

# Select the first 100 samples
num_samples = 100
if len(dataset) < num_samples:
    print(f"Warning: Dataset has only {len(dataset)} samples. Using all of them.")
    num_samples = len(dataset)

subset_dataset = dataset.select(range(num_samples))

# Create a DataLoader
batch_size = 16 # Adjust based on your GPU memory

# Lists to store data for DataFrame
results_data = {
    "batch_index": [],
    "token_position": [],
    "kl_divergence": [],
    "loss_base": [],
    "loss_dpo": [],
    "true_token": [],
    "pred_token_base": [],
    "pred_token_dpo": [],
    "context": [],
}

# Determine pad token id (adjust if necessary)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
print(f"Using Pad Token ID: {pad_token_id}")


print(f"Calculating KL divergence and Loss for {num_samples} samples...")
with torch.no_grad():
    # Use enumerate to get batch index
    for batch_idx in tqdm(range(0, num_samples, batch_size)):
        # Ensure we don't go beyond num_samples
        end_idx = min(batch_idx + batch_size, num_samples)
        if batch_idx >= end_idx: continue # Skip if batch is empty

        tensors = [torch.tensor(elem) for elem in subset_dataset[batch_idx:end_idx]['input_ids']]
        # Handle potential empty list if end_idx == batch_idx
        if not tensors: continue
        input_ids = torch.stack(tensors).to(device)
        # print(input_ids.shape) # Debug print
        # Target tokens are shifted versions of input_ids
        target_ids = input_ids[:, 1:].contiguous() # Shape: (batch_size, seq_len-1)

        # Create attention mask to ignore padding tokens
        # Shape: (batch_size, seq_len)
        attention_mask = (input_ids != pad_token_id).long()
        # Mask for target tokens (1 to seq_len-1)
        mask = attention_mask[:, 1:].bool() # Shape: (batch_size, seq_len-1), use bool for indexing

        # Get logits from both models
        # Shape: (batch_size, seq_len, vocab_size)
        logits_base = base_model(input_ids) # Ensure we get logits if output is not just tensor
        logits_dpo = dpo_model(input_ids)  # Ensure we get logits if output is not just tensor
        # print(f"Logits base shape: {logits_base.shape}") # Debug print
        # print(f"Logits dpo shape: {logits_dpo.shape}") # Debug print

        # Slice logits for KL/Loss calculation (exclude last token prediction)
        logits_base_kl = logits_base[:, :-1, :] # (batch, seq_len-1, vocab)
        logits_dpo_kl = logits_dpo[:, :-1, :] # (batch, seq_len-1, vocab)

        # --- Loss Calculation ---
        vocab_size = logits_base_kl.shape[-1]
        logits_base_for_loss = logits_base_kl.reshape(-1, vocab_size)
        logits_dpo_for_loss = logits_dpo_kl.reshape(-1, vocab_size)
        target_ids_for_loss = target_ids.reshape(-1)

        loss_base_per_token_flat = F.cross_entropy(logits_base_for_loss, target_ids_for_loss, reduction='none')
        loss_dpo_per_token_flat = F.cross_entropy(logits_dpo_for_loss, target_ids_for_loss, reduction='none')

        loss_base_per_token = loss_base_per_token_flat.view(input_ids.shape[0], -1) # (batch, seq_len-1)
        loss_dpo_per_token = loss_dpo_per_token_flat.view(input_ids.shape[0], -1) # (batch, seq_len-1)

        # --- KL Divergence Calculation ---
        log_probs_base = F.log_softmax(logits_base_kl, dim=-1) # (batch, seq_len-1, vocab)
        log_probs_dpo = F.log_softmax(logits_dpo_kl, dim=-1) # (batch, seq_len-1, vocab)

        # Ensure kl_div calculation handles potential NaNs or Infs if necessary
        kl_div_per_token = F.kl_div(log_probs_dpo, log_probs_base, log_target=True, reduction='none').sum(dim=-1) # (batch, seq_len-1)
        kl_div_per_token = torch.nan_to_num(kl_div_per_token) # Replace NaN/Inf

        # --- Collect data for DataFrame ---
        # Iterate through sequences in the batch
        for i in range(input_ids.shape[0]):
            # Iterate through token positions (0 to seq_len-2)
            # Ensure j does not exceed sequence length dimensions
            seq_len_minus_1 = logits_base_kl.shape[1] # Use the actual dimension size
            for j in range(seq_len_minus_1):
                # Ensure mask indices are within bounds
                if j < mask.shape[1] and mask[i, j].item(): # Check if it's NOT a padding token position
                    # Get true token ID
                    true_token_id = target_ids[i, j].item()
                    true_token = tokenizer.convert_ids_to_tokens([true_token_id])[0] # Use convert_ids_to_tokens

                    # Get predicted token IDs
                    pred_token_id_base = torch.argmax(logits_base_kl[i, j, :]).item()
                    pred_token_base = tokenizer.convert_ids_to_tokens([pred_token_id_base])[0]

                    pred_token_id_dpo = torch.argmax(logits_dpo_kl[i, j, :]).item()
                    pred_token_dpo = tokenizer.convert_ids_to_tokens([pred_token_id_dpo])[0]

                    # Get context
                    # Position in the original input_ids corresponds to j+1
                    current_pos = j + 1
                    context_window = 5
                    start_ctx = max(0, current_pos - context_window)
                    end_ctx = min(input_ids.shape[1], current_pos + context_window + 1) # +1 to include current token in decode? Check context logic

                    # Extract context IDs, excluding padding
                    context_ids_raw = input_ids[i, start_ctx:end_ctx].tolist()
                    # Filter out pad tokens if needed, but decoding handles them usually
                    # context_ids = [tok_id for tok_id in context_ids_raw if tok_id != pad_token_id]

                    context_text = tokenizer.decode(context_ids_raw, skip_special_tokens=True) # Decode the raw context ids


                    results_data["batch_index"].append(batch_idx + i) # Use global index
                    results_data["token_position"].append(j)
                    results_data["kl_divergence"].append(kl_div_per_token[i, j].item())
                    results_data["loss_base"].append(loss_base_per_token[i, j].item())
                    results_data["loss_dpo"].append(loss_dpo_per_token[i, j].item())
                    results_data["true_token"].append(true_token)
                    results_data["pred_token_base"].append(pred_token_base)
                    results_data["pred_token_dpo"].append(pred_token_dpo)
                    results_data["context"].append(context_text)


# --- Create DataFrame ---
results_df = pd.DataFrame(results_data)


print(results_df[results_df["kl_divergence"] > 0.1])

# --- Sort by KL Divergence ---
results_df = results_df.sort_values(by='kl_divergence', ascending=False)







print("\n--- Results DataFrame ---")
print(results_df.head()) # Print the first few rows
print(f"\nDataFrame shape: {results_df.shape}")

# Optional: Save the DataFrame to a CSV file
# results_df.to_csv("kl_loss_per_token.csv", index=False)
# print("\nDataFrame saved to kl_loss_per_token.csv")


# --- Optional: Calculate and print overall averages (as before) ---
if not results_df.empty:
    average_kl_div = results_df["kl_divergence"].mean()
    average_loss_base = results_df["loss_base"].mean()
    average_loss_dpo = results_df["loss_dpo"].mean()
    print(f"\nAverage KL Divergence (Base || DPO) per token: {average_kl_div:.4f}")
    print(f"Average Loss (Base Model) per token: {average_loss_base:.4f}")
    print(f"Average Loss (DPO Model) per token: {average_loss_dpo:.4f}")
else:
    print("\nNo valid tokens found to calculate KL divergence or loss.")





# %%
# --- Distribution of KL Divergence ---
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(results_df, x="kl_divergence", bins=100, kde=False)
plt.title("Distribution of KL Divergence")
plt.xlim(0, 1)
plt.xlabel("KL Divergence")
plt.ylabel("Frequency")
plt.savefig("experiments_alex/kl_divergence/kl_divergence_distribution.png")
plt.close()





