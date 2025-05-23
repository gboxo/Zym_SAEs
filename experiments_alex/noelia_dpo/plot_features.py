#!/usr/bin/env python
# Top-level script: plot_features.py

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

from datasets import load_from_disk
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# Helper to load an MSA from FASTA
# -----------------------------------------------------------------------------
def load_msa_fasta(path):
    """
    Load an MSA FASTA and return a list of aligned sequences (including gaps).
    """
    msa_seqs = []
    with open(path, "r") as f:
        current = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current:
                    msa_seqs.append(current)
                current = ""
            else:
                current += line
        if current:
            msa_seqs.append(current)
    return msa_seqs

# -----------------------------------------------------------------------------
# Helpers to map between original sequence indices and MSA indices
# -----------------------------------------------------------------------------
def map_positions_to_msa(positions, original_seq, msa_seq):
    """
    Map positions in the original (ungapped) sequence to positions in the MSA.
    positions: iterable of ints in [0..len(original_seq)-1]
    Returns a list of the same length, with -1 if that residue is aligned to a gap.
    """
    seq_to_msa = {}
    orig_i = 0
    for msa_i, ch in enumerate(msa_seq):
        if ch != '-':
            seq_to_msa[orig_i] = msa_i
            orig_i += 1

    msa_positions = []
    for pos in positions:
        msa_positions.append(seq_to_msa.get(pos, -1))
    return msa_positions

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load your precomputed features
    feat_pkl = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/features_M3/features/features_M0_D0.pkl"
    with open(feat_pkl, "rb") as f:
        features = pkl.load(f)
    feat_indices = [
        14080,
        4482,
        9225,
        9611,
        11406,
        10127,
        6288,
        14230,
        9755,
        3101,
        15011,
        1712,
        5692,
        3394,
        5316,
        6094,
        13263,
        7127,
        4192,
        12002,
        7914,
        13549,
        6396,
        2174
    ]



    # 2) Load original sequences (we assume the same dataset used to produce features)
    ds_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_3/eval/"
    tokenizer_path = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
    dataset = load_from_disk(ds_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    seqs = [tokenizer.decode(seq) for seq in dataset["input_ids"]]
    for tok in ["<sep>", "<start>", "<end>", "<pad>", "<|endoftext|>", " ", "3.2.1.1"]:
        seqs = [s.replace(tok, "") for s in seqs]

    # 3) Load the MSA
    msa_fasta = "/home/woody/b114cb/b114cb23/boxo/kl_divergence/msa_it3.fasta"
    msa_seqs = load_msa_fasta(msa_fasta)
    assert len(msa_seqs) == len(seqs) == len(features), "Mismatch # sequences"

    msa_len = len(msa_seqs[0])  # all rows in an MSA share the same length

    # 4) Precompute for padding originals
    keys = list(features.keys())
    orig_lens = [features[k].shape[0] for k in keys]
    max_orig_len = max(orig_lens)

    out_dir = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/features_M3/figures/"
    cmap = plt.cm.viridis
    cmap.set_bad("white")

    for feat_idx in feat_indices:
        all_msa  = []
        all_orig = []

        # Build per‐sequence MSA‐aligned and original arrays
        for i, key in enumerate(keys):
            mat  = features[key].todense()      # shape (L_i, F)
            arr  = np.array(mat)
            vals = arr[:, feat_idx]             # (L_i,)

            # 1) original
            all_orig.append(vals)

            # 2) MSA‐aligned
            orig_seq = seqs[i]
            msa_seq  = msa_seqs[i]
            mapping = map_positions_to_msa(range(len(orig_seq)), orig_seq, msa_seq)

            padded = np.full(msa_len, -1.0, dtype=float)
            for pos, msa_pos in enumerate(mapping):
                if msa_pos != -1 and pos < len(vals):
                    padded[msa_pos] = vals[pos]
            all_msa.append(padded)

        # Stack into matrices
        data_msa  = np.vstack(all_msa)   # (N_seq, msa_len)
        data_orig = np.full((len(keys), max_orig_len), -1.0, dtype=float)
        for i, vals in enumerate(all_orig):
            data_orig[i, : len(vals)] = vals

        # Mask values < 0.01 (this also masks our -1 padding)
        masked_msa  = np.ma.masked_where(data_msa  < 0.01, data_msa)
        masked_orig = np.ma.masked_where(data_orig < 0.01, data_orig)

        # Plot side-by-side
        fig, (ax0, ax1) = plt.subplots(
            1, 2, sharey=True, figsize=(16, 8),
            gridspec_kw={'width_ratios': [msa_len, max_orig_len]}
        )

        # Left: MSA‐aligned heatmap
        im0 = ax0.imshow(masked_msa, aspect='auto', cmap=cmap, interpolation='nearest')
        ax0.set_title(f"Feat #{feat_idx} (MSA‐aligned)")
        ax0.set_ylabel("Sequence")
        ax0.set_xlabel("MSA Position")
        num_m = data_msa.shape[1]
        step = max(1, num_m // 20)
        ticks = np.arange(0, num_m, step)
        ax0.set_xticks(ticks)

        # Right: Original heatmap
        im1 = ax1.imshow(masked_orig, aspect='auto', cmap=cmap, interpolation='nearest')
        ax1.set_title(f"Feat #{feat_idx} (Original)")
        ax1.set_xlabel("Sequence Position")
        num_o = data_orig.shape[1]
        step_o = max(1, num_o // 20)
        ticks_o = np.arange(0, num_o, step_o)
        ax1.set_xticks(ticks_o)
        ax1.set_xticklabels(ticks_o)

        # Shared colorbar
        #fig.colorbar(im0, ax=[ax0, ax1], label="Feature value", pad=0.02)

        plt.tight_layout()
        plt.savefig(f"{out_dir}feature_{feat_idx}_side_by_side.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
