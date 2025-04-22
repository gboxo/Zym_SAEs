import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu

def load_steering_data(file_path):
    with open(file_path, "r") as f:
        data = f.read()
        lines = data.split("\n")
    lines = [elem for elem in lines if elem != ""]
    ids = [seq for seq in lines if seq.startswith(">")]
    features = [int(seq.split(",")[1]) for seq in lines if not seq.startswith(">")]
    predictions = [float(seq.split(",")[2]) for seq in lines if not seq.startswith(">")]
    
    df = pd.DataFrame(list(zip(ids, features, predictions)), columns=["name", "feature", "prediction"])
    df.drop(columns=["name"], inplace=True)
    return df

def load_ablation_data(ablation_path, base_path):
    df = pd.read_csv(ablation_path)
    df.columns = ["name", "feature", "prediction"]
    df.drop(columns=["name"], inplace=True)
    
    df_base = pd.read_csv(base_path, sep="\t|,", header=None)
    df_base.columns = ["name", "prediction1", "prediction2"]
    df_base["prediction"] = (df_base["prediction1"] + df_base["prediction2"]) / 2
    df_base.drop(columns=["name", "prediction1", "prediction2"], inplace=True)
    
    return df, df_base

def create_violin_plot(df, iteration_num, output_dir, plot_type="ablation"):
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='feature', y='prediction', data=df)
    plt.xlabel('Feature')
    plt.ylabel('Prediction')
    plt.title('Violin Plot of Predictions by Feature')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/violin_plot_iteration{iteration_num}_{plot_type}.png")
    plt.close()

def create_spike_plot(df, df_base, iteration_num, output_dir, plot_type="ablation"):
    plt.figure(figsize=(10, 6))
    df_mean = df.groupby('feature').mean().sort_values(by='prediction', ascending=False)
    sns.barplot(x=np.arange(len(df_mean)), y='prediction', data=df_mean)
    plt.xlabel('Feature')
    plt.ylabel('Prediction')
    plt.title('Mean Predictions by Feature')
    plt.axhline(y=df_base['prediction'].mean(), color='red', linestyle='--', label='Base Prediction')
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/spike_plot_iteration{iteration_num}_{plot_type}.png")
    plt.close()

def perform_statistical_tests(df):
    # Pairwise Wilcoxon test
    unique_features = df['feature'].unique()
    wilcoxon_results = []
    for i in range(len(unique_features)):
        for j in range(i+1, len(unique_features)):
            feature_1 = df[df['feature'] == unique_features[i]]['prediction']
            feature_2 = df[df['feature'] == unique_features[j]]['prediction']
            min_length = min(feature_1.shape[0], feature_2.shape[0])
            
            stat, p_value = wilcoxon(feature_1[:min_length], feature_2[:min_length])
            wilcoxon_results.append({
                'feature_1': unique_features[i],
                'feature_2': unique_features[j],
                'stat': stat,
                'p_value': p_value
            })
    
    # Mann-Whitney U test (one-against-all)
    mannwhitney_results = []
    for feature in unique_features:
        feature_1 = df[df['feature'] == feature]['prediction']
        feature_2 = df[df['feature'] != feature]['prediction']
        
        stat, p_value = mannwhitneyu(feature_1, feature_2)
        mannwhitney_results.append({
            'feature': feature,
            'stat': stat,
            'p_value': p_value
        })
    
    return pd.DataFrame(wilcoxon_results), pd.DataFrame(mannwhitney_results)

def main():
    output_dir = "/home/woody/b114cb/b114cb23/boxo/Diffing_Analysis_Data/latent_scoring_figures"
    
    # Process iterations
    for iteration_num in range(1, 6):
        base_path = f"/home/woody/b114cb/b114cb23/boxo/activity_predictions/activity_prediction_iteration{iteration_num-1}.txt"
        ablation_path = f"/home/woody/b114cb/b114cb23/boxo/activity_predictions_ablation/activity_prediction_iteration{iteration_num}.txt"
        steering_path = f"/home/woody/b114cb/b114cb23/boxo/activity_predictions_steering/activity_prediction_iteration{iteration_num}.txt"
        
        # Process ablation data
        df_ablation, df_base = load_steering_data(ablation_path), load_steering_data(base_path)
        create_violin_plot(df_ablation, iteration_num, output_dir, "ablation")
        create_spike_plot(df_ablation, df_base, iteration_num, output_dir, "ablation")
        
        # Process steering data
        df_steering = load_steering_data(steering_path)
        create_violin_plot(df_steering, iteration_num, output_dir, "steering")
        create_spike_plot(df_steering, df_base, iteration_num, output_dir, "steering")
        
        # Perform statistical tests
        wilcoxon_results, mannwhitney_results = perform_statistical_tests(df_ablation)
        print(f"\nIteration {iteration_num} Results:")
        print("\nWilcoxon Test Results:")
        print(wilcoxon_results)
        print("\nMann-Whitney U Test Results:")
        print(mannwhitney_results)

if __name__ == "__main__":
    main()






