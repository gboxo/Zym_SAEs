import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import warnings
import os
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings('ignore')





# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_attribution_results(filepath):
    """Load attribution results from pickle file"""
    with open(filepath, "rb") as f:
        position_results = pickle.load(f)
    return position_results

def create_layer_specific_heatmap(position_results, layer, pdf_pages):
    """Create a heatmap showing top feature attributions for a specific layer"""
    
    positions = list(position_results.keys())
    
    # Get top K features for this layer across all positions
    K = 20  # Top 20 features
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data for this layer
    layer_data = []
    position_labels = []
    
    for pos in positions:
        if position_results[pos][layer]['attributions'] is not None:
            attributions = position_results[pos][layer]['attributions'].squeeze()
            if attributions.dim() > 1:
                attributions = attributions.mean(dim=0)  # Average across sequences
            
            # Get top K features
            top_values, top_indices = torch.topk(attributions, K)
            layer_data.append(top_values.numpy())
            
            transition = position_results[pos][layer]['transition']
            position_labels.append(f"Pos {pos}\n{transition[0]}→{transition[1]}")
    
    if layer_data:
        # Create heatmap for this layer
        data_matrix = np.array(layer_data)
        
        im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
        ax.set_title(f'Layer {layer.split(".")[1]} - Top {K} Feature Attributions', fontsize=14)
        ax.set_ylabel('Position/Transition')
        ax.set_xlabel('Top Features (ranked)')
        ax.set_yticks(range(len(position_labels)))
        ax.set_yticklabels(position_labels)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    pdf_pages.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_layer_specific_barplot(position_results, layer, pdf_pages):
    """Create bar plots showing top contributing features for a specific layer"""
    
    positions = list(position_results.keys())
    
    # Create subplots for each position in this layer
    n_positions = len(positions)
    cols = min(3, n_positions)
    rows = (n_positions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_positions == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for pos_idx, pos in enumerate(positions):
        row = pos_idx // cols
        col = pos_idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        if position_results[pos][layer]['attributions'] is not None:
            attributions = position_results[pos][layer]['attributions'].squeeze()
            if attributions.dim() > 1:
                attributions = attributions.mean(dim=0)  # Average across sequences
            
            # Get top 10 features
            top_values, top_indices = torch.topk(attributions, 10)
            
            # Create bar plot
            bars = ax.bar(range(10), top_values.numpy())
            transition = position_results[pos][layer]['transition']
            ax.set_title(f'Pos {pos}: {transition[0]}→{transition[1]}')
            ax.set_xlabel('Top Features')
            ax.set_ylabel('Attribution Score')
            ax.set_xticks(range(10))
            ax.set_xticklabels([f'F{idx.item()}' for idx in top_indices], rotation=45)
            
            # Color bars by value
            colors = plt.cm.RdYlBu_r(top_values.numpy() / top_values.max().item())
            for bar, color in zip(bars, colors):
                bar.set_color(color)
        else:
            ax.set_title(f'Pos {pos}: No Data')
            ax.axis('off')
    
    # Hide unused subplots
    for pos_idx in range(n_positions, rows * cols):
        row = pos_idx // cols
        col = pos_idx % cols
        if rows > 1:
            axes[row, col].axis('off')
        elif cols > 1:
            axes[col].axis('off')
    
    fig.suptitle(f'Layer {layer.split(".")[1]} - Top 10 Features by Position', fontsize=16)
    plt.tight_layout()
    pdf_pages.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_layer_statistics_plot(position_results, layer, pdf_pages):
    """Create statistics plots for a specific layer"""
    
    positions = list(position_results.keys())
    
    # Calculate summary statistics for this layer
    summary_data = []
    
    for pos in positions:
        if position_results[pos][layer]['attributions'] is not None:
            attributions = position_results[pos][layer]['attributions'].squeeze()
            if attributions.dim() > 1:
                attributions = attributions.mean(dim=0)
            
            # Calculate statistics
            max_attr = attributions.max().item()
            mean_attr = attributions.mean().item()
            std_attr = attributions.std().item()
            num_active = (attributions > 0.01).sum().item()
            
            transition = position_results[pos][layer]['transition']
            
            summary_data.append({
                'Position': pos,
                'Transition': f"{transition[0]}→{transition[1]}",
                'Max Attribution': max_attr,
                'Mean Attribution': mean_attr,
                'Std Attribution': std_attr,
                'Active Features': num_active
            })
    
    if not summary_data:
        return
    
    df = pd.DataFrame(summary_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Max attribution by position
    bars1 = axes[0,0].bar(df['Position'], df['Max Attribution'])
    axes[0,0].set_title('Max Attribution by Position')
    axes[0,0].set_xlabel('Position')
    axes[0,0].set_ylabel('Max Attribution')
    for i, (pos, trans) in enumerate(zip(df['Position'], df['Transition'])):
        axes[0,0].text(i, df['Max Attribution'].iloc[i] + 0.01, trans, 
                      ha='center', va='bottom', rotation=45, fontsize=8)
    
    # Number of active features
    bars2 = axes[0,1].bar(df['Position'], df['Active Features'], color='orange')
    axes[0,1].set_title('Number of Active Features (>0.01)')
    axes[0,1].set_xlabel('Position')
    axes[0,1].set_ylabel('Active Features')
    
    # Mean vs Std attribution
    scatter = axes[1,0].scatter(df['Mean Attribution'], df['Std Attribution'], 
                               c=df['Position'], cmap='viridis', s=100)
    axes[1,0].set_xlabel('Mean Attribution')
    axes[1,0].set_ylabel('Std Attribution')
    axes[1,0].set_title('Mean vs Std Attribution')
    plt.colorbar(scatter, ax=axes[1,0], label='Position')
    
    # Attribution distribution
    metrics = ['Max Attribution', 'Mean Attribution', 'Active Features']
    positions_list = df['Position'].tolist()
    x = np.arange(len(positions_list))
    width = 0.25
    
    for i, metric in enumerate(metrics[:3]):  # Only plot 3 metrics
        if metric == 'Active Features':
            # Normalize active features to same scale as attributions
            values = df[metric] / df[metric].max() * df['Max Attribution'].max()
        else:
            values = df[metric]
        axes[1,1].bar(x + i*width, values, width, 
                     label=metric, alpha=0.8)
    
    axes[1,1].set_xlabel('Position')
    axes[1,1].set_ylabel('Normalized Values')
    axes[1,1].set_title('Comparison of Metrics by Position')
    axes[1,1].set_xticks(x + width)
    axes[1,1].set_xticklabels(positions_list)
    axes[1,1].legend()
    
    fig.suptitle(f'Layer {layer.split(".")[1]} - Statistical Summary', fontsize=16)
    plt.tight_layout()
    pdf_pages.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_layer_feature_analysis(position_results, layer, pdf_pages):
    """Create detailed feature analysis for a specific layer"""
    
    positions = list(position_results.keys())
    
    # Get top features across all positions for this layer
    all_top_features = set()
    position_features = {}
    
    for pos in positions:
        if position_results[pos][layer]['attributions'] is not None:
            attributions = position_results[pos][layer]['attributions'].squeeze()
            if attributions.dim() > 1:
                attributions = attributions.mean(dim=0)
            
            top_values, top_indices = torch.topk(attributions, 20)
            top_features_set = set(top_indices.numpy())
            all_top_features.update(top_features_set)
            position_features[pos] = {
                'indices': top_indices.numpy(),
                'values': top_values.numpy(),
                'attributions': attributions
            }
    
    if not all_top_features:
        return
    
    # Create feature overlap matrix
    overlap_matrix = np.zeros((len(positions), len(positions)))
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            if pos1 in position_features and pos2 in position_features:
                features1 = set(position_features[pos1]['indices'][:10])  # Top 10
                features2 = set(position_features[pos2]['indices'][:10])  # Top 10
                overlap = len(features1 & features2)
                overlap_matrix[i, j] = overlap
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature overlap heatmap
    im1 = axes[0].imshow(overlap_matrix, cmap='Blues')
    axes[0].set_title('Feature Overlap Between Positions\n(Top 10 features)')
    axes[0].set_xlabel('Position')
    axes[0].set_ylabel('Position') 
    axes[0].set_xticks(range(len(positions)))
    axes[0].set_xticklabels(positions)
    axes[0].set_yticks(range(len(positions)))
    axes[0].set_yticklabels(positions)
    
    # Add text annotations
    for i in range(len(positions)):
        for j in range(len(positions)):
            axes[0].text(j, i, f'{int(overlap_matrix[i, j])}', 
                        ha='center', va='center', color='white' if overlap_matrix[i, j] > 5 else 'black')
    
    plt.colorbar(im1, ax=axes[0])
    
    # Feature importance distribution
    feature_importance = {}
    for pos in positions:
        if pos in position_features:
            for idx, val in zip(position_features[pos]['indices'], position_features[pos]['values']):
                if idx not in feature_importance:
                    feature_importance[idx] = []
                feature_importance[idx].append(val)
    
    # Get features that appear in multiple positions
    multi_position_features = {k: v for k, v in feature_importance.items() if len(v) > 1}
    
    if multi_position_features:
        # Plot distribution of important features
        feature_ids = list(multi_position_features.keys())[:10]  # Top 10 multi-position features
        feature_means = [np.mean(multi_position_features[fid]) for fid in feature_ids]
        feature_stds = [np.std(multi_position_features[fid]) for fid in feature_ids]
        
        x_pos = np.arange(len(feature_ids))
        axes[1].bar(x_pos, feature_means, yerr=feature_stds, capsize=5, alpha=0.7)
        axes[1].set_title('Multi-Position Feature Importance\n(Mean ± Std)')
        axes[1].set_xlabel('Feature ID')
        axes[1].set_ylabel('Attribution Score')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'F{fid}' for fid in feature_ids], rotation=45)
    else:
        axes[1].text(0.5, 0.5, 'No multi-position features found', 
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Multi-Position Feature Analysis')
    
    fig.suptitle(f'Layer {layer.split(".")[1]} - Feature Analysis', fontsize=16)
    plt.tight_layout()
    pdf_pages.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to generate layer-specific PDFs"""
    
    # Path to the attribution results
    results_path = "/home/woody/b114cb/b114cb23/boxo/dpo_noelia/attributions.pkl"
    out_dir = "/home/woody/b114cb/b114cb23/boxo/circuit_analysis/attribution_plots/"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading attribution results...")
    position_results = load_attribution_results(results_path)
    
    # Get all layers
    positions = list(position_results.keys())
    if positions:
        layers = list(position_results[positions[0]].keys())
    else:
        print("No position data found!")
        return
    
    print(f"Found {len(layers)} layers: {[layer.split('.')[1] for layer in layers]}")
    
    # Create separate PDF for each layer
    for layer in layers:
        layer_num = layer.split('.')[1]
        pdf_filename = f"{out_dir}layer_{layer_num}_attribution_analysis.pdf"
        
        print(f"Creating PDF for Layer {layer_num}...")
        
        with PdfPages(pdf_filename) as pdf_pages:
            # Page 1: Layer-specific heatmap
            print(f"  - Creating heatmap for Layer {layer_num}")
            create_layer_specific_heatmap(position_results, layer, pdf_pages)
            
            # Page 2: Layer-specific bar plots
            print(f"  - Creating bar plots for Layer {layer_num}")
            create_layer_specific_barplot(position_results, layer, pdf_pages)
            
            # Page 3: Layer statistics
            print(f"  - Creating statistics for Layer {layer_num}")
            create_layer_statistics_plot(position_results, layer, pdf_pages)
            
            # Page 4: Feature analysis
            print(f"  - Creating feature analysis for Layer {layer_num}")
            create_layer_feature_analysis(position_results, layer, pdf_pages)
            
            # Add metadata to PDF
            d = pdf_pages.infodict()
            d['Title'] = f'Layer {layer_num} Attribution Analysis'
            d['Author'] = 'SAE Attribution Analysis'
            d['Subject'] = f'Feature attribution analysis for layer {layer_num}'
            d['Keywords'] = f'SAE, Attribution, Layer{layer_num}, Features'
            d['Creator'] = 'Python matplotlib'
    
    # Also create a summary CSV with all data
    print("Creating summary CSV...")
    summary_data = []
    for pos in positions:
        for layer in layers:
            if position_results[pos][layer]['attributions'] is not None:
                attributions = position_results[pos][layer]['attributions'].squeeze()
                if attributions.dim() > 1:
                    attributions = attributions.mean(dim=0)
                
                max_attr = attributions.max().item()
                mean_attr = attributions.mean().item()
                std_attr = attributions.std().item()
                num_active = (attributions > 0.01).sum().item()
                
                transition = position_results[pos][layer]['transition']
                
                summary_data.append({
                    'Position': pos,
                    'Layer': int(layer.split('.')[1]),
                    'Transition': f"{transition[0]}→{transition[1]}",
                    'Max Attribution': max_attr,
                    'Mean Attribution': mean_attr,
                    'Std Attribution': std_attr,
                    'Active Features': num_active
                })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(out_dir + "attribution_summary.csv", index=False)
    
    print(f"\nAll layer-specific PDFs saved to {out_dir}")
    print(f"Summary statistics saved to {out_dir}attribution_summary.csv")
    print(f"\nCreated PDFs:")
    for layer in layers:
        layer_num = layer.split('.')[1]
        print(f"  - layer_{layer_num}_attribution_analysis.pdf")

if __name__ == "__main__":
    main()
