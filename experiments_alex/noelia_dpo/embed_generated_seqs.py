import os
import re
import glob
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Bio import SeqIO
from sklearn.manifold import TSNE

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def load_sequences(seq_dir):
    """
    Scan seq_dir for any .fasta files whose basename contains
    'iteration<NUMBER>.fasta' (with arbitrary prefixes), parse
    all FASTA records via a custom parser (to avoid SeqIO),
    and return a DataFrame with columns [seq_id, iteration, sequence].
    """
    # match any filename ending in 'iteration<NUM>.fasta'
    pattern = re.compile(r"iteration(\d+)\.fasta$")
    # pick up all .fasta files, then filter to those matching our pattern
    all_fasta = glob.glob(os.path.join(seq_dir, "*.fasta"))
    fasta_paths = [p for p in all_fasta if pattern.search(os.path.basename(p))]
    # sort by the captured iteration number (numerical order)
    fasta_paths.sort(key=lambda p: int(pattern.search(os.path.basename(p)).group(1)))

    records = []
    for path in fasta_paths:
        iteration = int(pattern.search(os.path.basename(path)).group(1))
        with open(path, "r") as f:
            # split on '>' to get each record block (first split is empty)
            for block in f.read().split(">"):
                if not block:
                    continue
                lines = block.splitlines()
                header = lines[0]
                seq_lines = lines[1:]
                # take only the ID (ignore any numeric flags or comments)
                seq_id = header.split()[0]
                # join all sequence lines into one continuous string
                seq = "".join(seq_lines).strip()
                if not seq:
                    continue
                records.append({
                    "seq_id": seq_id,
                    "iteration": iteration,
                    "sequence": seq
                })
    print(records)

    return pd.DataFrame(records)

        
        


def get_last_token_embeddings(df, model, tokenizer, device):
    """
    For each row in df, tokenize the sequence, run the model
    (with hidden_states), and extract the last‐token
    activation at every layer. Returns a dict keyed by
    (seq_id, iteration) -> np.array(shape=(n_layers, hidden_dim)).
    """
    embeddings = {}
    model.eval()
    # wrap the DataFrame iterator with tqdm
    for idx, row in tqdm(df.iterrows(),
                        total=len(df),
                        desc="Embedding sequences"):
        seq_id    = row["seq_id"]
        iteration = row["iteration"]
        seq       = row["sequence"]

        # Tokenize and move to device
        inputs = tokenizer(seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # Stack last-token activations from each layer
        layer_embs = torch.stack([
            hs[0, -1, :].cpu() for hs in hidden_states
        ], dim=0)  # shape = (n_layers, hidden_dim)

        embeddings[(seq_id, iteration)] = layer_embs.numpy()

    return embeddings


def save_embeddings(embeddings, out_path):
    """Pickle the embeddings dict to out_path."""
    with open(out_path, "wb") as f:
        pickle.dump(embeddings, f)


def tsne_and_plot(embeddings, df, output_dir):
    """
    For each layer, run t-SNE on the last-token embeddings
    of all sequences, color by iteration, and plot in a grid.
    """
    # Prepare
    seq_keys = list(embeddings.keys())
    n_layers = embeddings[seq_keys[0]].shape[0]
    iterations = [k[1] for k in seq_keys]
    palette = sns.color_palette("tab10", df["iteration"].nunique())

    # Compute t-SNE per layer
    tsne_results = {}
    for layer in range(n_layers):
        X = np.stack([embeddings[k][layer] for k in seq_keys])
        tsne = TSNE(n_components=2, random_state=42)
        coords = tsne.fit_transform(X)
        tsne_results[layer] = coords

    # Plot grid
    n_cols = 4
    n_rows = math.ceil(n_layers / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for layer in range(n_layers):
        ax = axes[layer]
        coords = tsne_results[layer]
        sns.scatterplot(
            x=coords[:, 0],
            y=coords[:, 1],
            hue=iterations,
            palette=palette,
            legend=False,
            ax=ax
        )
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Turn off any extra axes
    for ax in axes[n_layers:]:
        ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "tsne_last_token_grid.png")
    plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    # Paths
    model_dir = "/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
    seq_dir = "/home/woody/b114cb/b114cb23/DPO_amylase_run_SAPI_FT_v2"
    output_dir = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia"
    os.makedirs(output_dir, exist_ok=True)

    # Device, tokenizer, model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
    model = AutoModel.from_pretrained(model_dir).to(device)

    # Load sequences
    df = load_sequences(seq_dir)

    # Get embeddings
    embeddings = get_last_token_embeddings(df, model, tokenizer, device)

    # Save embeddings
    pkl_path = os.path.join(output_dir, "last_token_layer_embeddings.pkl")
    save_embeddings(embeddings, pkl_path)

    # Run t-SNE and plot
    tsne_and_plot(embeddings, df, output_dir)


if __name__ == "__main__":
    main()
