from diffing_utils import load_config
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
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from scipy.interpolate import griddata
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


def get_important_features(X,pred1,pred2,plddt,tm_score):
    """
    Fit a sparse logistic regression probe to the data, to select the most important features
    """

    # Prediction 1
    X_train, X_test, y_train, y_test = train_test_split(X,pred1>3)
    results_pred1, probes_pred1 = fit_lr_probe(X_train,y_train, X_test, y_test)
    # Prediction 2
    X_train, X_test, y_train, y_test = train_test_split(X,pred2>1.5)
    results_pred2, probes_pred2 = fit_lr_probe(X_train,y_train, X_test, y_test)
    # PLDDT 
    X_train, X_test, y_train, y_test = train_test_split(X,plddt>0.7)
    results_plddt, probes_plddt = fit_lr_probe(X_train,y_train, X_test, y_test)
    # TM-score
    X_train, X_test, y_train, y_test = train_test_split(X,tm_score>0.8)
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








def get_correlations(mean_features, plddt, activity, tm_score, f_rates, cs, output_dir):
    """Calculate correlations between features and metrics."""
    # Calculate correlations between features and metrics
    correlations = []
    p_values = []
    
    for i in range(mean_features.shape[1]):
        feature = mean_features[:, i]
        if np.std(feature) > 0:
        
        # Calculate correlations with each metric
            corr_plddt, p_plddt = pearsonr(feature, plddt)
            corr_activity, p_activity = pearsonr(feature, activity)
            corr_tm, p_tm = pearsonr(feature, tm_score)
            
            # Store the correlations and p-values
            correlations.append([corr_plddt, corr_activity, corr_tm])
            p_values.append([p_plddt, p_activity, p_tm])
        else:
            correlations.append([0,0,0])
            p_values.append([0,0,0])
    
    correlations = np.array(correlations)
    p_values = np.array(p_values)
    
    # Calculate mean absolute correlation for each feature
    mean_abs_corr = np.mean(np.abs(correlations), axis=1)
    
    # Get the top features by mean absolute correlation
    top_k = 25
    top_indices = np.argsort(mean_abs_corr)[-top_k:][::-1]
    top_correlations = correlations[top_indices]
    top_p_values = p_values[top_indices]
    
    # Apply Benjamini-Hochberg correction for multiple testing
    mask = multipletests(p_values.flatten(), method='fdr_bh')[0].reshape(p_values.shape)
    
    # Create a correlation data dictionary
    correlation_data = {
        'feature_indices': top_indices,
        'correlations': correlations,
        'p_values': p_values,
        'significant': ~mask,
        'mean_abs_corr': mean_abs_corr,
        'f_rates': f_rates,
        'cs': cs
    }
    
    # Save the correlation data
    os.makedirs(f"{output_dir}/correlations", exist_ok=True)
    pkl.dump(correlation_data, open(f"{output_dir}/correlations/top_correlations_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    
    print(f"Created correlation data with shape: {top_correlations.shape}")
    print(f"Number of significant correlations after correction: {np.sum(~mask)}")
    
    return correlation_data


def plot_correlation_heatmap(correlation_data, output_dir):
    """Plot a heatmap of the top correlations."""
    correlations = correlation_data['correlations']
    top_indices = np.abs(correlations).sum(axis=1).argsort()[-100:][::-1]

    top_correlations = correlations[top_indices]
    
    plt.figure(figsize=(10, 10))
    sns.heatmap(top_correlations, cmap="coolwarm", center=0, 
                xticklabels=["pLDDT", "Activity", "TM-score"])
    plt.title("Top Feature Correlations (masked by significance)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/correlation_heatmap_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()


def plot_firing_rate_vs_correlation(correlation_data, output_dir):
    """Plot firing rates vs mean absolute correlation."""
    f_rates = correlation_data['f_rates']
    correlations = correlation_data['correlations']
    top_indices = np.abs(correlations).sum(axis=1).argsort()[-100:][::-1]
    plt.figure(figsize=(10, 10))
    plt.scatter(f_rates[top_indices], correlations[top_indices,1], cmap="coolwarm")
    plt.xlabel("Firing Rates")
    plt.ylabel("Activity")
    plt.title("Firing Rates vs Activity")
    plt.savefig(f"{output_dir}/figures/firing_rates_activity_M{model_iteration}_D{data_iteration}.png", dpi=300)
    plt.close()


def plot_2d_density(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_dir):
    """Create a 2D density plot for CS and firing rate, colored by mean absolute correlation."""
    # Get data
    
    # Remove any NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Create a grid for the 2D density estimation
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)
    
    # Create a kernel density estimate for x and y
    xy_data = np.vstack([x, y])
    kde = gaussian_kde(xy_data)
    
    # Evaluate the density on the grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    density = kde(positions).reshape(X.shape)
    
    # Create a 2D interpolation of z values (mean absolute correlation)
    Z = griddata((x, y), z, (X, Y), method='linear', fill_value=np.nan)
    
    # Create the interactive 2D plot
    fig = go.Figure()
    
    # Add the density contours
    fig.add_trace(go.Contour(
        x=xi, y=yi, z=density,
        contours=dict(
            showlabels=True,
            labelfont=dict(size=12, color='white')
        ),
        colorscale='Blues',
        showscale=False,
        opacity=0.5,
        name='Density'
    ))
    
    # Add the mean absolute correlation as a colored heatmap
    fig.add_trace(go.Heatmap(
        x=xi, y=yi, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Mean Absolute Correlation'),
        opacity=0.8,
        name='Mean Abs Correlation'
    ))
    
    # Update layout
    fig.update_layout(
        title='2D Density Plot: Firing Rates vs CS (colored by Mean Absolute Correlation)',
        xaxis_title='Firing Rates',
        yaxis_title='CS Values',
        width=900,
        height=700
    )
    
    # Save as interactive HTML
    fig.write_html(f"{output_dir}/figures/2d_density_plot_interactive_M{model_iteration}_D{data_iteration}.html")
    
    # Also save a static image for reference
    fig.write_image(f"{output_dir}/figures/2d_density_plot_M{model_iteration}_D{data_iteration}.png", scale=2)


def plot_3d_density(correlation_data, output_dir):
    """Create an interactive 3D density plot."""
    # Get data
    x = correlation_data['f_rates']  # Firing rates
    y = correlation_data['cs']       # CS values
    z = correlation_data['mean_abs_corr']  # Mean absolute correlation
    
    # Remove any NaN or infinite values
    valid_indices = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[valid_indices]
    y = y[valid_indices]
    z = z[valid_indices]
    
    # Create a grid of points for the density estimation
    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    zi = np.linspace(z.min(), z.max(), 50)
    
    # Create 3D grid
    X, Y = np.meshgrid(xi, yi)
    
    # Create a kernel density estimate for all three dimensions
    xyz_data = np.vstack([x, y, z])
    kde = gaussian_kde(xyz_data)
    
    # For each x,y point, find the z value with maximum density
    Z = np.zeros_like(X)
    density_values = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Create a line of points along z-axis at this x,y coordinate
            points = np.vstack([np.full(len(zi), X[i,j]), 
                               np.full(len(zi), Y[i,j]), 
                               zi])
            # Evaluate density along this line
            density_along_z = kde(points)
            # Find z with maximum density
            max_idx = np.argmax(density_along_z)
            Z[i,j] = zi[max_idx]
            density_values[i,j] = density_along_z[max_idx]
    
    # Create the interactive 3D surface plot
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=density_values,
        colorscale='Viridis',
        colorbar=dict(title='Density')
    )])
    
    # Update layout
    fig.update_layout(
        title='Interactive 3D Density Surface: Firing Rates, CS, and Mean Absolute Correlation',
        scene=dict(
            xaxis_title='Firing Rates',
            yaxis_title='CS Values',
            zaxis_title='Mean Absolute Correlation',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800
    )
    
    # Save as interactive HTML
    os.makedirs(f"{output_dir}/figures", exist_ok=True)
    fig.write_html(f"{output_dir}/figures/3d_density_plot_interactive_M{model_iteration}_D{data_iteration}.html")
    
    # Also save a static image for reference
    fig.write_image(f"{output_dir}/figures/3d_density_plot.png", scale=2)


def analyze_correlations(mean_features, plddt, activity, tm_score, f_rates, cs):
    """Main function to analyze correlations and create visualizations."""
    # Calculate correlations and get data
    correlation_data = get_correlations(mean_features, plddt, activity, tm_score, f_rates, cs)
    
    # Create various plots
    plot_correlation_heatmap(correlation_data, output_dir)
    plot_firing_rate_vs_correlation(correlation_data, output_dir)
    plot_2d_density(correlation_data['f_rates'], correlation_data['cs'], correlation_data['correlations'][:,1], output_dir)
    plot_3d_density(correlation_data, output_dir)
    
    return correlation_data

def analyze_important_features(mean_features, activity, activity2, plddt, tm_score):
    """Main function to analyze important features and create visualizations."""
    # Calculate correlations and get data
    unique_coefs, coefs = get_important_features(mean_features, activity, activity2, plddt, tm_score)
    
    # Create various plots
    
    importance_features = {
        'unique_coefs': unique_coefs,
        'coefs': coefs,
    }
    
    # Save the correlation data
    os.makedirs(f"{output_dir}/important_features", exist_ok=True)
    pkl.dump(importance_features, open(f"{output_dir}/important_features/important_features_M{model_iteration}_D{data_iteration}.pkl", "wb"))
    

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

    analyze_correlations(mean_features, plddt, activity, tm_score, f_rates, cs, output_dir)
    analyze_important_features(mean_features, activity, activity2, plddt, tm_score, output_dir)


