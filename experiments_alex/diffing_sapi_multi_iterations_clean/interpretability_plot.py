import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch.nn.functional as F

# Set the device for computation
device = torch.device('cuda')

# Define directories for model checkpoints
model_dirs = [f'/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration{i}' for i in range(1, 31)]

# Load reference model and tokenizer
ref_model = AutoModelForCausalLM.from_pretrained('/home/woody/b114cb/b114cb23/models/ZymCTRL').to(device)
tokenizer = AutoTokenizer.from_pretrained('/home/woody/b114cb/b114cb23/models/ZymCTRL')

# Define output file title
output_title = 'heatmap_3.2.1.1_Mut_plddt_iteration30.pdf'


# Input text
text = ['3.2.1.1<sep><start>MSRFVTSALLLALLFLAFASANAALSAAEWRSQSIYQVIIDRMYSDSTTTAACNTTAYCGGTWQGIINQLDYIQQMGFTAIQISPIIKNIFGSPFYAQYFHPFNLNSAFHGEA<end><|endoftext|>']
all_r_values_list = []
titles = []
tokens_list = []

def log_likelihood(sequence, device, model, tokenizer):
    """
    Compute log-likelihood of a sequence using the given model and tokenizer.
    """
    model.eval()
    encodings = tokenizer(sequence, return_tensors='pt').to(device)
    outputs = model(**encodings)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    input_ids = encodings.input_ids
    token_log_probs = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs

# Compute log-likelihood for the reference model
sequence = text[0]
ref_log_prob = log_likelihood(sequence, device, ref_model, tokenizer)

# Initialize lists for storing results
all_r_values_list = []
titles = []

# Iterate over models and compute r values
for idx, model_dir in enumerate(model_dirs):
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    log_prob = log_likelihood(sequence, device, model, tokenizer)
    inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
    input_ids = inputs[0]
    tokens = tokenizer.decode(input_ids.cpu()).split()
    tokens_list.append(tokens)

    beta = 0.001
    r = beta * (log_prob - ref_log_prob)
    r_values = r.squeeze(0).cpu().tolist()
    
    all_r_values_list.append(r_values)
    iteration = model_dir.split('/')[-1].split('iteration')[1]
    titles.append(f'Iteration {iteration}')
    
    print(f"Processed: Iteration {iteration}")
    print(r_values)
    print(sum(r_values))
    del model
    torch.cuda.empty_cache()

# Convert r values to a DataFrame
r_matrix = np.array(all_r_values_list)
df = pd.DataFrame(r_matrix, index=titles)

# Determine color range for heatmap
all_r_values_flat = r_matrix.flatten()
vmin, vmax = np.min(all_r_values_flat), np.max(all_r_values_flat)

# Highlight specific columns
highlight_columns = [7, 10, 41, 42, 43, 44, 45, 66, 85]  # Binding pocket columns
highlight_columns = [x + 8 for x in highlight_columns]

highlight_columns2 = [12, 119]  # Active site columns
highlight_columns2 = [x + 8 for x in highlight_columns2]

# Configure figure dimensions
num_iterations = len(df.index)
row_height, margin = 0.3, 10
figure_height = num_iterations * row_height + margin

num_tokens = len(df.columns)
column_width = 0.5
figure_width = num_tokens * column_width + margin

# Plot heatmap
plt.figure(figsize=(figure_width, figure_height))
sns.set(font_scale=0.8)
ax = sns.heatmap(
    df,
    cmap='seismic',
    xticklabels=True,
    yticklabels=False,
    vmin=vmin,
    vmax=vmax,
    cbar=False,
    linewidths=3,
    linecolor='white'
)

# Customize x-axis labels
tokens = tokens_list[0]  # Example, assuming all sequences have the same tokens
ax.set_xticklabels(tokens, rotation=90, fontsize=30)

# Highlight specific columns
x_labels = ax.get_xticklabels()
for idx, label in enumerate(x_labels):
    if idx in highlight_columns2:
        label.set_bbox(dict(facecolor='red', alpha=0.5, edgecolor='none', pad=1))
    elif idx in highlight_columns:
        label.set_bbox(dict(facecolor='orange', alpha=0.5, edgecolor='none', pad=1))
    else:
        label.set_bbox(dict(facecolor='none', edgecolor='none', pad=1))

# Add labels and title
plt.xlabel('Tokens')
plt.ylabel('Model Iteration')
plt.title('Changes in r over Tokens and Iterations')

# Save and display the plot
plt.tight_layout()
plt.savefig(output_title)