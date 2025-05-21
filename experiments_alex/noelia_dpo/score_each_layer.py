import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load_embeddings(pkl_path):
    """
    Load the dict mapping (seq_id, iteration) -> np.array(shape=(n_layers, hidden_dim))
    """
    with open(pkl_path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings

def score_each_layer(embeddings, cv=5):
    """
    For each layer in the embeddings, train a one‐vs‐all LogisticRegression
    (with cv‐fold cross‐validation) and return a DataFrame of accuracies.
    """
    # Get a stable list of keys and stack into array
    keys = list(embeddings.keys())
    # embeddings[k] has shape (n_layers, hidden_dim)
    n_layers = embeddings[keys[0]].shape[0]
    # Build X_all: shape = (n_samples, n_layers, hidden_dim)
    X_all = np.stack([embeddings[k] for k in keys], axis=0)
    # Build labels y_all from the iteration part of each key
    y_all = np.array([k[1] for k in keys])


    results = []
    for layer in range(n_layers):
        print(layer)
        X = X_all[:, layer, :]

        clf = LogisticRegression(
            multi_class="ovr",
            solver="lbfgs",
            max_iter=1000,
        )
        # 5‐fold (by default stratified) cross‐validation
        scores = cross_val_score(clf, X, y_all, cv=cv, scoring="accuracy")
        print(scores)
        results.append({
            "layer": layer,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std(),
        })

    return pd.DataFrame(results)

def main():
    # Path to your pickled embeddings
    pkl_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/last_token_layer_embeddings.pkl"
    # or wherever you saved it in embed_generated_seqs.py

    # Load
    embeddings = load_embeddings(pkl_path)

    # Score
    df_scores = score_each_layer(embeddings, cv=5)

    # Print table
    print("\nPer‐Layer Logistic Regression Accuracy (5‐fold CV):\n")
    print(df_scores.to_string(index=False, float_format="%.3f"))

    # Optionally save to CSV
    out_csv = os.path.splitext(pkl_path)[0] + "_layer_accuracies.csv"
    df_scores.to_csv(out_csv, index=False)
    print(f"\nSaved per‐layer accuracies to {out_csv}")

if __name__ == "__main__":
    main()
