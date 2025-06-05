import pickle as pkl
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

# Load tokenizer - adjust the model name to match the one used in your project
tokenizer = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/model-3.2.1.1/")

kl_divergence_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_kl_divergence.pkl"
base_log_probs_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_base_log_probs.pkl"
dpo_log_probs_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_dpo_log_probs.pkl"

with open(kl_divergence_path, "rb") as f:
    kl_divergence = pkl.load(f)

with open(base_log_probs_path, "rb") as f:
    base_log_probs = pkl.load(f)

with open(dpo_log_probs_path, "rb") as f:
    dpo_log_probs = pkl.load(f)

    
    

base = torch.stack([torch.tensor(base_log_probs[i]) for i in range(len(base_log_probs))])
dpo = torch.stack([torch.tensor(dpo_log_probs[i]) for i in range(len(dpo_log_probs))])
kl = torch.stack([torch.tensor(elem) for elem in list(kl_divergence.values())])

# Helper function to decode token indices to text
def decode_tokens(token_indices):
    return [tokenizer.decode(idx) for idx in token_indices]

# Create DataFrame to store results
results = []

# Process each sequence
for seq_id in range(kl.shape[0]):
    # Get top 10 positions with highest KL divergence
    kl_seq = kl[seq_id]
    top_positions = torch.topk(kl_seq, k=10, dim=0)
    positions = top_positions.indices.tolist()
    kl_values = top_positions.values.tolist()
    
    for rank, (position, kl_value) in enumerate(zip(positions, kl_values)):
        # Get log probs for base and DPO at this position
        base_log_prob = base[seq_id, 0, position, :]
        dpo_log_prob = dpo[seq_id, 0, position, :]
        
        # Get top 3 tokens by probability
        base_top3 = torch.topk(base_log_prob, k=3, dim=0)
        dpo_top3 = torch.topk(dpo_log_prob, k=3, dim=0)
        
        base_top3_indices = base_top3.indices.tolist()
        dpo_top3_indices = dpo_top3.indices.tolist()
        
        # Get max token by log prob
        base_max_token = torch.argmax(base_log_prob).item()
        dpo_max_token = torch.argmax(dpo_log_prob).item()
        
        # Decode tokens to text
        base_top3_text = decode_tokens(base_top3_indices)
        dpo_top3_text = decode_tokens(dpo_top3_indices)
        base_max_text = tokenizer.decode(base_max_token)
        dpo_max_text = tokenizer.decode(dpo_max_token)
        
        # Check if max token is the same
        is_same = base_max_token == dpo_max_token
        
        # Format top 3 tokens with both index and text
        base_top3_str = [f"{idx} ({text})" for idx, text in zip(base_top3_indices, base_top3_text)]
        dpo_top3_str = [f"{idx} ({text})" for idx, text in zip(dpo_top3_indices, dpo_top3_text)]
        
        # Add to results
        results.append({
            'Seq_id': seq_id,
            'Top': rank + 1,  # 1-indexed rank
            'Position': position,
            'KL': kl_value,
            'Base top 3': base_top3_str,
            'DPO top 3': dpo_top3_str,
            'Base max': f"{base_max_token} ({base_max_text})",
            'DPO max': f"{dpo_max_token} ({dpo_max_text})",
            'is_the_same': is_same
        })

# Create DataFrame
df = pd.DataFrame(results)

# Display the DataFrame
print(df.sort_values('KL', ascending=False).head(20))
# Display the DataFrame
df2= df[df['Position']==285]
print(df2.sort_values('KL', ascending=False).head(50))

# Optionally save to CSV
df.to_csv('kl_divergence_analysis.csv', index=False)

# Create a new dataframe grouped by position
position_stats = []

# Group by position
for position, group in df.groupby('Position'):
    # Count entries with this position
    all_count = len(group)
    
    # Calculate average KL divergence
    avg_kl = group['KL'].mean()
    
    # Calculate proportion of same max tokens
    same_token_pct = (group['is_the_same'].sum() / all_count) * 100
    
    # Find most common transitions
    transitions = []
    for _, row in group.iterrows():
        if not row['is_the_same']:
            base_max = row['Base max']
            dpo_max = row['DPO max']
            transitions.append((base_max, dpo_max))
    
    # Count transitions
    transition_counts = {}
    for t in transitions:
        if t in transition_counts:
            transition_counts[t] += 1
        else:
            transition_counts[t] = 1
    
    # Get top 3 transitions
    top_transitions = []
    if transition_counts:
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        for (base, dpo), count in sorted_transitions[:3]:
            top_transitions.append(f"{base} → {dpo} ({count} times)")
    
    # Add to results
    position_stats.append({
        'Position': position,
        'Count': all_count,
        'Avg KL': avg_kl,
        'Same Token %': same_token_pct,
        'Top Transitions': top_transitions
    })

# Create position stats DataFrame
position_df = pd.DataFrame(position_stats)

# Sort by average KL divergence
position_df = position_df.sort_values('Count', ascending=False)
# Remove entries with same token 100
position_df = position_df[position_df['Same Token %'] != 100]

# Display the position stats DataFrame
print("\nPosition Statistics:")
print(position_df.head(50))

# Optionally save to CSV
position_df.to_csv('position_transition_analysis.csv', index=False)






