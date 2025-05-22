import pandas as pd
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer
import pickle
import torch

"""
We need to remove 9 positions to make up for the prompt prefix

In [5]: tokenizer("3.2.1.1<sep><start>")["input_ids"]
Out[5]: [9, 431, 8, 431, 7, 431, 7, 2, 3]
"""


def map_seq_to_msa(original_seq, msa_seq):
    """
    Maps positions from the original sequence to the MSA sequence.
    
    Args:
        original_seq (str): The original sequence without gaps
        msa_seq (str): The MSA sequence with gaps (represented by '-')
        
    Returns:
        list: A mapping array where each index corresponds to an MSA position.
              The value is the corresponding position in the original sequence,
              or -1 if the MSA position is a gap.
    """
    mapping = []
    orig_pos = 0
    
    for msa_char in msa_seq:
        if msa_char == '-':
            mapping.append(-1)
        else:
            # Verify the characters match (sanity check)
            if orig_pos < len(original_seq) and original_seq[orig_pos] != msa_char:
                print(f"Warning: Character mismatch at position {orig_pos}. "
                      f"Original: {original_seq[orig_pos]}, MSA: {msa_char}")
            mapping.append(orig_pos)
            orig_pos += 1
    
    return mapping

def map_positions_to_msa(positions, original_seq, msa_seq):
    """
    Maps an array of positions from the original sequence to MSA positions.
    
    Args:
        positions (list): List of positions in the original sequence
        original_seq (str): The original sequence without gaps
        msa_seq (str): The MSA sequence with gaps
        
    Returns:
        list: Positions mapped to the MSA coordinate system, with -1 for unmapped positions
    """
    # Create the mapping from original sequence to MSA
    seq_to_msa_map = {}
    orig_pos = 0
    
    for msa_pos, msa_char in enumerate(msa_seq):
        if msa_char != '-':
            seq_to_msa_map[orig_pos] = msa_pos
            orig_pos += 1
    
    # Map each position using the mapping
    msa_positions = []
    for pos in positions:
        msa_positions.append(seq_to_msa_map.get(pos, -1))
    
    return msa_positions

def load_and_process_kl_divergences(kl_path, seqs, msa_seqs, prompt_length=9):
    """
    Loads KL divergences, removes prompt tokens and right padding.
    
    Args:
        kl_path (str): Path to the KL divergence pickle file
        seqs (list): Original sequences without gaps
        msa_seqs (list): MSA sequences with gaps
        prompt_length (int): Number of tokens to remove from the beginning (prompt)
        
    Returns:
        list: Processed KL divergences aligned with MSA positions
    """
    # Load KL divergences using the same method as in plot_kl.py
    with open(kl_path, "rb") as f:
        kl_divergences = pickle.load(f)
    kl_divergences = list(kl_divergences.values())
    kl_divergences = torch.tensor(kl_divergences)
    kl_divergences = kl_divergences.detach().cpu().numpy()
    
    # Process each sequence and its KL divergence
    processed_kls = []
    
    for i, (orig_seq, msa_seq, kl_div) in enumerate(zip(seqs, msa_seqs, kl_divergences)):
        # Remove prompt tokens
        kl_div = kl_div[prompt_length:]
        
        # Determine right padding based on sequence length
        seq_length = len(orig_seq)
        # The remaining length after removing prompt tokens
        valid_length = seq_length
        
        # Trim the KL divergence to the valid length
        kl_div = kl_div[:valid_length]
        
        # Map the KL divergence to MSA positions
        msa_kl = np.ones(len(msa_seq)) * -1  # Initialize with -1
        
        # Create mapping from original positions to MSA positions
        orig_to_msa_positions = map_positions_to_msa(list(range(len(orig_seq))), orig_seq, msa_seq)
        
        # Assign KL values to corresponding MSA positions
        for orig_pos, msa_pos in enumerate(orig_to_msa_positions):
            if msa_pos != -1 and orig_pos < len(kl_div):
                msa_kl[msa_pos] = kl_div[orig_pos]
        
        processed_kls.append(msa_kl)
    
    return processed_kls

# Example usage:
if __name__ == "__main__":
    # Path to KL divergence data
    kl_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/kl_divergence/M3_kl_divergence.pkl"
    
    dataset = load_from_disk("/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_3/eval/")



    tokenizer = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/model-3.2.1.1/")

    input_ids = dataset['input_ids']

    seqs = [tokenizer.decode(seq) for seq in input_ids]

    special_tokens = ["3.2.1.1<sep><start>", "<sep>", "<start>", "<end>", "<pad>", " "]

    # Remove the special tokens from the sequences
    seqs = [seq.replace("<sep>","") for seq in seqs]
    seqs = [seq.replace("<start>","") for seq in seqs]
    seqs = [seq.replace("<end>","") for seq in seqs]
    seqs = [seq.replace("<pad>","") for seq in seqs]
    seqs = [seq.replace("<|endoftext|>","") for seq in seqs]
    seqs = [seq.replace(" ","") for seq in seqs]
    seqs = [seq.replace("3.2.1.1","") for seq in seqs]




    if False:
        with open("/home/woody/b114cb/b114cb23/boxo/kl_divergence/sequences_it3.fasta", "w") as f:
            for i, seq in enumerate(seqs):
                f.write(f">sequence_{i}\n{seq}\n")
        
        
    path = "/home/woody/b114cb/b114cb23/boxo/kl_divergence/msa_it3.fasta"

    # Load MSA sequences
    msa_seqs = []
    with open(path, "r") as f:
        current_seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    msa_seqs.append(current_seq)
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            msa_seqs.append(current_seq)

    
    # Load and process KL divergences
    processed_kls = load_and_process_kl_divergences(kl_path, seqs, msa_seqs)
    processed_kls = np.array(processed_kls)
    
    # Create two masks - one for -1 values and one for small values
    mask_neg1 = processed_kls == -1
    mask_small = processed_kls < 0.1
    
    # Create a copy of the data that we'll modify
    masked_kl = processed_kls.copy()
    
    # Set values < 0.01 to black (0 in viridis colormap)
    masked_kl = np.ma.masked_array(masked_kl, mask=mask_small)
    
    # Set -1 values to white by masking them
    masked_kl = np.ma.masked_array(masked_kl, mask=mask_neg1)
    
    import matplotlib.pyplot as plt

    # Calculate average KL divergence per position, ignoring masked values
    print(masked_kl.shape)
    print(masked_kl)
    avg_kl_per_pos = np.mean(masked_kl, axis=0)
    print(avg_kl_per_pos.shape)
    print(avg_kl_per_pos)
    avg_kl_reshaped = avg_kl_per_pos.reshape(1, -1)  # Reshape to 2D for imshow
    
    # Mask positions in avg_kl_reshaped if most of the column in masked_kl is masked
    # Calculate the fraction of masked values per column in masked_kl
    # masked_kl.mask is True where values are masked
    if hasattr(masked_kl, 'mask') and masked_kl.mask is not np.ma.nomask:
        fraction_masked_per_column = np.sum(masked_kl.mask, axis=0) / masked_kl.shape[0]
    else: # If nothing is masked in masked_kl initially
        fraction_masked_per_column = np.zeros(masked_kl.shape[1])

    # Define a threshold for masking the average plot
    masking_threshold = 0.9  # If >90% of values in a column are masked
    
    # Create a mask for avg_kl_reshaped
    avg_mask = fraction_masked_per_column > masking_threshold
    
    # Apply the mask to avg_kl_reshaped
    # Ensure avg_mask is broadcastable to avg_kl_reshaped.data if it's a plain numpy array
    # If avg_kl_reshaped is already a masked array, combine masks
    if isinstance(avg_kl_reshaped, np.ma.MaskedArray):
        avg_kl_reshaped.mask = np.logical_or(avg_kl_reshaped.mask, avg_mask.reshape(1, -1))
    else:
        avg_kl_reshaped = np.ma.masked_array(avg_kl_reshaped, mask=avg_mask.reshape(1, -1))

    avg_kl_reshaped[avg_kl_reshaped < 0.01] = np.ma.masked # Also apply original threshold, ensuring masked values are handled

    # Create figure with 2 rows, shared x–axis, custom height ratios
    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 10),
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.05}
    )

    cmap = plt.cm.viridis
    cmap.set_bad('white')  # -1 values will be white

    # Top subplot: full heatmap
    im0 = ax0.imshow(
        masked_kl,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=0.01
    )
    # Attach a single colorbar to both axes so they shrink equally
    cbar = fig.colorbar(im0, ax=[ax0, ax1], label='KL Divergence', pad=0.02)
    ax0.set_title('KL Divergence Heatmap')
    ax0.set_ylabel('Sequence Number')
    ax0.set_xticklabels([])  # hide x labels on top plot

    # Decide tick positions (at most ~20 ticks)
    num_positions = masked_kl.shape[1]
    tick_interval = max(1, num_positions // 20)
    tick_positions = list(range(0, num_positions, tick_interval))
    ax0.set_xticks(tick_positions)

    # Bottom subplot: average KL per position
    im1 = ax1.imshow(
        avg_kl_reshaped,
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        vmin=0.01
    )
    ax1.set_yticks([])  # no y–ticks on the bottom bar
    ax1.set_xlabel('MSA Position')
    ax1.set_title('Average KL Divergence per Position')
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_positions)

    # Final layout adjustments
    plt.tight_layout()
    plt.savefig('kl_divergence_combined_M3.png', dpi=300, bbox_inches='tight')

    # Save the processed KL divergences
    np.save("/home/woody/b114cb/b114cb23/boxo/kl_divergence/processed_kl_divergences.npy", processed_kls)
