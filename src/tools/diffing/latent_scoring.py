from src.tools.diffing.diffing_utils import load_config
import pandas as pd
import argparse
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import pickle as pkl
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

def get_activations( model, tokenizer, sequence):
    sequence = "3.2.1.1<sep><start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("25.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.25.hook_resid_pre"]
    return activations

def get_features(sae: JumpReLUSAE, activations):
    feature_acts = sae.forward(activations, use_pre_enc_bias=True)["feature_acts"]


    sparse_feature_acts = coo_matrix(feature_acts[0].detach().cpu().numpy())
    del feature_acts
    torch.cuda.empty_cache()
    return sparse_feature_acts


def get_all_features(model, sae, tokenizer, sequences):
    all_features = []
    for sequence in tqdm(sequences):
        activations = get_activations(model, tokenizer, sequence)
        features = get_features(sae, activations)
        all_features.append(features)
        del activations, features
        torch.cuda.empty_cache()
    return all_features

def obtain_features(df, output_dir):
    """
    Obtain features from natural sequences
    """
    sequences = df["sequence"].tolist()
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    pkl.dump(features, open(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    del features
    torch.cuda.empty_cache()

def load_features(path):
    """
    Load features from a file
    """
    assert path.endswith(".pkl"), "File must end with .pkl"
    features = pkl.load(open(path, "rb"))
    return features

def get_mean_features(features):
    """
    Get the mean features
    """
    mean_features = []
    for feature in features:
        mean_features.append(feature.todense()[10:].sum(axis=0))
    mean_features = np.array(mean_features)
    return mean_features



def firing_rates(features, output_dir):
    """
    Get the firing rates of the features

    1) Average number of firings per sequence with at least one firing
    2) Percentage of tokens that fire at least once per sequence
    3) Average number of firings per token
    """
    firing_rates_seq = []
    for feature in features:
        feats = feature.todense()[10:].sum(axis=0)
        w = np.where(feats > 0, 1, 0)
        fa = w.sum(axis=0)>0
        firing_rates_seq.append(fa)
    firing_rates_seq = np.array(firing_rates_seq).mean(axis=0)
    return firing_rates_seq


def fit_lr_probe(X_train, y_train, X_test, y_test):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    """
    results = []
    probes =[]
    for sparsity in tqdm(np.logspace(-4.5, -3, 10)):
        lr_model = LogisticRegressionCV(cv=5, penalty="l1", solver="liblinear", class_weight="balanced", Cs=[sparsity], n_jobs=-1)
        lr_model.fit(X_train, y_train)
        coefs = lr_model.coef_
        active_features = np.where(coefs != 0)[1]
        probes.append(lr_model)
        y_pred = lr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        roc_auc = roc_auc_score(y_test, y_pred)
        results.append({
            "active_features": active_features,
            "sparsity": sparsity,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
        })
    return results, probes


def get_important_features(X, pred1, pred2, plddt, tm_score, thresholds, directions='upper'):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    
    Args:
        X: Input features
        pred1, pred2, plddt, tm_score: Metrics to analyze
        thresholds: dict with keys 'pred1', 'pred2', 'plddt', 'tm_score' containing threshold values
        directions: str or dict, either 'upper' or 'lower' for all metrics, or dict with keys for each metric
    """
    # Standardize directions if string
    if isinstance(directions, str):
        directions = {k: directions for k in ['pred1', 'pred2', 'plddt', 'tm_score']}

    def get_mask(values, threshold, direction):
        return values > threshold if direction == 'upper' else values < threshold

    def get_assert(y_train, y_test):
        """
        Assert that y_train and y_test are have more than one class
        """
        assert len(np.unique(y_train)) > 1, "y_train must have more than one class"
        assert len(np.unique(y_test)) > 1, "y_test must have more than one class"

    def process_lr_probe(X, pred, thresholds, directions):
        """
        Process the LR probe
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, get_mask(pred, thresholds, directions))
        get_assert(y_train, y_test)
        results, probes = fit_lr_probe(X_train, y_train, X_test, y_test)
        return results, probes


    # Prediction 1
    results_pred1, probes_pred1 = process_lr_probe(X, pred1, thresholds['pred1'], directions['pred1'])

    # Prediction 2
    results_pred2, probes_pred2 = process_lr_probe(X, pred2, thresholds['pred2'], directions['pred2'])

    # PLDDT 
    results_plddt, probes_plddt = process_lr_probe(X, plddt, thresholds['plddt'], directions['plddt'])

    # TM-score
    results_tm_score, probes_tm_score = process_lr_probe(X, tm_score, thresholds['tm_score'], directions['tm_score'])


    coefs_pred1 = torch.tensor(probes_pred1[-1].coef_)[0]
    coefs_pred2 = torch.tensor(probes_pred2[-1].coef_)[0]
    coefs_plddt = torch.tensor(probes_plddt[-1].coef_)[0]
    coefs_tm_score = torch.tensor(probes_tm_score[-1].coef_)[0]

    unique_coefs = torch.unique(torch.cat([torch.where(coefs_pred1>0)[0],
                                        torch.where(coefs_pred2>0)[0],
                                        torch.where(coefs_plddt>0)[0],
                                        torch.where(coefs_tm_score>0)[0]]))
    coefs = torch.stack([coefs_pred1, coefs_pred2, coefs_plddt, coefs_tm_score])
    coefs = coefs[:,unique_coefs]

    return unique_coefs, coefs







def analyze_important_features(mean_features, activity, activity2, plddt, tm_score, thresholds, direction):
    """Main function to analyze important features and create visualizations."""
    # Calculate correlations and get data


    unique_coefs, coefs = get_important_features(mean_features, activity, activity2, plddt, tm_score, thresholds, direction)
    
    # Create various plots
    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': coefs,
    }
    # Save the correlation data
    return importance_features
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str)
    args = parser.parse_args()
    cfg_path = args.cfg_path
    config = load_config(cfg_path)
    iteration_num = config["iteration_num"]
    model_iteration = config["model_iteration"]
    data_iteration = config["data_iteration"]
    ec_label = config["label"]
    cs_path = config["paths"]["cs_path"]
    df_path = config["paths"]["df_path"]
    output_dir = config["paths"]["output_dir"]
    model_path = config["paths"]["model_path"].format(iteration_num)
    sae_path = config["paths"]["sae_path"].format(model_iteration, data_iteration)
    disc_thresholds = config["thresholds"]




    cs = torch.load(cs_path)
    cs = cs[f"M{model_iteration}_D{data_iteration}_vs_M0_D0"].cpu().numpy()

    
    
    # Create the directories
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    os.makedirs(f"{output_dir}/features", exist_ok=True)



    
    

    assert os.path.exists(df_path), "Dataframe does not exist"
    df = pd.read_csv(df_path)
    if True:
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_50.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        # Load model
        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        torch.cuda.empty_cache()

        obtain_features(df, output_dir)

    features = load_features(f"{output_dir}/features/features_M{model_iteration}_D{data_iteration}.pkl")
    f_rates = firing_rates(features, output_dir)

    # Plot the histogram of the firing rates
    plt.hist(f_rates,bins=20)
    plt.savefig(f"{output_dir}/figures/firing_rates_histogram_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()



    mean_features = get_mean_features(features)[:,0]
    plddt = df["pLDDT"].tolist()
    plddt = np.array(plddt)

    activity = df["prediction1"].tolist()
    activity = np.array(activity)

    activity2 = df["prediction2"].tolist()
    activity2 = np.array(activity2)

    tm_score = df["alntmscore"].tolist()
    tm_score = np.array(tm_score)

    def get_thresholds_and_directions(thresholds):
        thresholds_pos = {}
        thresholds_neg = {}
        for key, value in thresholds.items():
            thresholds_pos[key] = value["upper"]
            thresholds_neg[key] = value["lower"]
        return thresholds_pos, thresholds_neg



    thresholds_pos, thresholds_neg = get_thresholds_and_directions(disc_thresholds)



    importance_features_pos = analyze_important_features(mean_features, activity, activity2, plddt, tm_score, thresholds_pos, "upper")
    importance_features_neg = analyze_important_features(mean_features, activity, activity2, plddt, tm_score, thresholds_neg, "lower")


    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    pkl.dump(importance_features_pos, open(f"{output_dir}/important_features/important_features_pos_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    pkl.dump(importance_features_neg, open(f"{output_dir}/important_features/important_features_neg_M{model_iteration}_D{data_iteration}.pkl", "wb"))
