import pandas as pd
import os
from src.inference.inference_batch_topk import convert_to_jumprelu
from src.utils import load_sae, load_model, get_ht_model
from src.training.sae import JumpReLUSAE
import torch
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm
import pickle as pkl
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split


def get_activations( model, tokenizer, sequence):
    sequence = "3.2.1.1<sep><start>" + sequence
    inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")
    with torch.no_grad():
        names_filter = lambda x: x.endswith("26.hook_resid_pre")
        _, cache = model.run_with_cache(inputs, names_filter=names_filter)
        activations = cache["blocks.26.hook_resid_pre"]
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

def obtain_features(df):
    """
    Obtain features from natural sequences
    """
    sequences = df["sequence"].tolist()
    features = get_all_features(model,jump_relu, tokenizer, sequences)
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features_32", exist_ok=True)
    pkl.dump(features, open(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features_32/features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
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



def firing_rates(features):
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
    np.save(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/firing_rates_M{model_iteration}_D{data_iteration}.npy", firing_rates_seq)
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


def get_important_features(X,pred1,pred2,plddt,tm_score, direction):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    """
    if direction == "pos":

        # Prediction 1
        X_train, X_test, y_train, y_test = train_test_split(X,pred1>1.5)
        results_pred1, probes_pred1 = fit_lr_probe(X_train,y_train, X_test, y_test)
        # Prediction 2
        X_train, X_test, y_train, y_test = train_test_split(X,pred2>1.7)
        results_pred2, probes_pred2 = fit_lr_probe(X_train,y_train, X_test, y_test)
        # PLDDT 
        X_train, X_test, y_train, y_test = train_test_split(X,plddt>0.8)
        results_plddt, probes_plddt = fit_lr_probe(X_train,y_train, X_test, y_test)
        # TM-score
        X_train, X_test, y_train, y_test = train_test_split(X,tm_score>0.75)
        results_tm_score, probes_tm_score = fit_lr_probe(X_train,y_train, X_test, y_test)
    elif direction == "neg":
        # Prediction 1
        X_train, X_test, y_train, y_test = train_test_split(X,pred1<1.1)
        results_pred1, probes_pred1 = fit_lr_probe(X_train,y_train, X_test, y_test)
        # Prediction 2
        X_train, X_test, y_train, y_test = train_test_split(X,pred2<1.1)
        results_pred2, probes_pred2 = fit_lr_probe(X_train,y_train, X_test, y_test)
        # PLDDT 
        X_train, X_test, y_train, y_test = train_test_split(X,plddt<0.5)
        results_plddt, probes_plddt = fit_lr_probe(X_train,y_train, X_test, y_test)
        # TM-score
        X_train, X_test, y_train, y_test = train_test_split(X,tm_score<0.5)
        results_tm_score, probes_tm_score = fit_lr_probe(X_train,y_train, X_test, y_test)
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





    return correlation_data

def analyze_important_features(mean_features, activity, activity2, plddt, tm_score):
    """Main function to analyze important features and create visualizations."""
    # Calculate correlations and get data
    unique_coefs, coefs = get_important_features(mean_features, activity, activity2, plddt, tm_score,direction="pos")
    
    # Create various plots
    
    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': coefs,
    }
    
    # Save the correlation data
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/important_features_32", exist_ok=True)
    pkl.dump(importance_features, open(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/important_features_32/pos_important_features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    unique_coefs, coefs = get_important_features(mean_features, activity, activity2, plddt, tm_score,direction="neg")
    
    # Create various plots
    
    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': coefs,
    }
    pkl.dump(importance_features, open(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/important_features_32/neg_important_features_M{model_iteration}_D{data_iteration}.pkl", "wb"))

    
    

if __name__ == "__main__":
    

    iteration_num = 0
    ec_label = "3.2.1.1"


    data_iteration = iteration_num 
    model_iteration = 0
    # Load the dataframe
    df_path = f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/dataframe_iteration{data_iteration}.csv"
    # Create the directories
    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features_32", exist_ok=True)
    assert os.path.exists(df_path), "Dataframe does not exist"
    df = pd.read_csv(df_path)
    if False:
        model_path = "/home/woody/b114cb/b114cb23/models/ZymCTRL/"
 
        sae_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/New_SAE/sae_training_iter_0_32/final/"
        cfg, sae = load_sae(sae_path)
        thresholds = torch.load(sae_path+"/percentiles/feature_percentile_99.pt")
        thresholds = torch.where(thresholds > 0, thresholds, torch.inf)
        sae.to("cuda")
        jump_relu = convert_to_jumprelu(sae, thresholds)
        jump_relu.eval()
        del sae
        # Load model
        tokenizer, model = load_model(model_path)
        model = get_ht_model(model, model.config).to("cuda")
        torch.cuda.empty_cache()

        obtain_features(df)

    os.makedirs(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features_32", exist_ok=True)
    features = load_features(f"/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/features_32/features_M{model_iteration}_D{data_iteration}.pkl")
    f_rates = firing_rates(features)

    mean_features = get_mean_features(features)[:,0]
    plddt = df["pLDDT"].tolist()
    plddt = np.array(plddt)

    activity = df["prediction1"].tolist()
    activity = np.array(activity)

    activity2 = df["prediction2"].tolist()
    activity2 = np.array(activity2)

    tm_score = df["alntmscore"].tolist()
    tm_score = np.array(tm_score)

    analyze_important_features(mean_features, activity, activity2, plddt, tm_score)


