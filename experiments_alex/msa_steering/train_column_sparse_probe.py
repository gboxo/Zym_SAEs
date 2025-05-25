import os
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def load_reindexed_features(path):
    """
    Load the sparse reindexed MSA features from a pickle file.
    Returns: dict {seq_key: {msa_pos: sparse_row_vector or None}}
    """
    with open(path, 'rb') as f:
        return pickle.load(f)

def train_column_sparse_probe(
    reindexed_features,
    labels,
    threshold=2.5,
    test_size=0.2,
    random_state=42
):
    """
    For each MSA column, train an L1‐penalized logistic probe.
    
    Args:
        reindexed_features: dict of {seq_key: {msa_pos: sparse_vector or None}}
        labels:            dict of {seq_key: float_prediction2}
        threshold:         float; binarization cutoff for prediction2
        test_size:         fraction of data in test split
        random_state:      seed for reproducibility
        
    Returns:
        coefs:      dict {msa_pos: np.ndarray of shape (D,)}
        intercepts: dict {msa_pos: float}
        metrics:    dict {msa_pos: {'auc': float, 'accuracy': float, 'f1': float, 'active_features': int}}
    """
    coefs = {}
    intercepts = {}
    metrics = {}

    # determine the total number of columns
    max_col = max(
        msa_pos
        for feats in reindexed_features.values()
        for msa_pos in feats.keys()
    )
    
    for col in range(max_col + 1):
        X_parts, y_parts = [], []
        for seq_key, feats in reindexed_features.items():
            if seq_key not in labels:
                continue
            vec = feats.get(col)
            if vec is None:
                continue
            X_parts.append(vec)
            y_parts.append(labels[seq_key])
        
        if len(y_parts) < 2:
            print(f"Column {col}: only {len(y_parts)} samples → skipping")
            continue
        
        # build data and binarize
        X = sparse.vstack(X_parts)              # (N × D)
        y = (np.array(y_parts) > threshold).astype(int)
        
        # stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                stratify=y,
                test_size=test_size,
                random_state=random_state
            )
        except ValueError as e:
            print(f"Column {col}: train_test_split error ({e}) → skipping")
            continue
        
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"Column {col}: class‐imbalance after split → skipping")
            continue
        
        # train L1 logistic probe
        model = LogisticRegressionCV(
            penalty='l1',
            solver='liblinear',
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            fit_intercept=False,
            Cs = [6e-3],
            cv=5,
            n_jobs=-1,
            scoring='f1'
            
        )
        model.fit(X_train, y_train)
        
        # evaluate
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)
        nz  = int(np.count_nonzero(model.coef_[0]))
        
        print(f"Column {col}: AUC = {auc:.3f}, Acc = {acc:.3f}, F1 = {f1:.3f}, Non-zero = {nz}")
        
        coefs[col]      = model.coef_[0]
        intercepts[col] = float(model.intercept_[0])
        metrics[col]    = {
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'active_features': nz
        }
    
    return coefs, intercepts, metrics

if __name__ == "__main__":
    # paths — adjust as needed
    FEATURES_PKL = "/home/woody/b114cb/b114cb23/boxo/msa_steering/reindexed_features_by_msa_sparse.pkl"
    CSV_PATH     = "/home/woody/b114cb/b114cb23/boxo/msa_steering/activity_predictions_no_penalty.csv"
    OUTPUT_PKL   = "/home/woody/b114cb/b114cb23/boxo/msa_steering/msa_column_probe_coeffs.pkl"

    # load data
    reindexed = load_reindexed_features(FEATURES_PKL)
    df        = pd.read_csv(CSV_PATH)
    if "prediction2" not in df.columns:
        raise KeyError("CSV must contain a 'prediction2' column")
    label_map = df.set_index("index")["prediction2"].to_dict()
    
    # train probes
    coefs, intercepts, metrics = train_column_sparse_probe(
        reindexed,
        label_map,
        threshold=2,
        test_size=0.2,
        random_state=42
    )
    
    # save coefficients + intercepts
    with open(OUTPUT_PKL, "wb") as outf:
        pickle.dump({"coefs": coefs, "intercepts": intercepts}, outf)
    print(f"→ Saved probe coefficients for {len(coefs)} columns to {OUTPUT_PKL}")

    # prepare performance dataframe
    df_metrics = pd.DataFrame.from_dict(metrics, orient='index')
    df_metrics.sort_index(inplace=True)
    
    base_dir = os.path.dirname(OUTPUT_PKL)
    
    # 1) Lineplot: AUC, Accuracy & F1 vs. MSA column
    plt.figure(figsize=(10, 4))
    plt.plot(df_metrics.index, df_metrics['auc'],      label='AUC',      marker='o')
    plt.plot(df_metrics.index, df_metrics['accuracy'], label='Accuracy', marker='s')
    plt.plot(df_metrics.index, df_metrics['f1'],       label='F1 Score', marker='^')
    plt.xlabel('MSA Column')
    plt.ylabel('Metric Value')
    plt.title('Probe Performance by MSA Column')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    out_path1 = os.path.join(base_dir, "msa_column_auc_acc_f1.png")
    plt.tight_layout()
    plt.savefig(out_path1, dpi=150)
    print(f"→ Saved performance lineplot (AUC, Acc, F1) to {out_path1}")
    plt.close()
    
    # 2) Barplot: Number of active (non-zero) features per column
    plt.figure(figsize=(10, 4))
    plt.bar(df_metrics.index, df_metrics['active_features'], color='C2')
    plt.xlabel('MSA Column')
    plt.ylabel('Active Features')
    plt.title('Non-zero Coefficients by MSA Column')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    out_path2 = os.path.join(base_dir, "msa_column_active_features.png")
    plt.tight_layout()
    plt.savefig(out_path2, dpi=150)
    print(f"→ Saved active-features barplot to {out_path2}")
    plt.close()
