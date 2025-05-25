import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm



for layer in [10, 15, 20, 25]:
    base_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/sae_{layer}/diffing/checkpoint_latest.pt"
    rl_path = f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2_layer_{layer}/diffing/checkpoint_latest.pt"

    base_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/sae_{layer}/diffing/thresholds.pt")
    rl_threshold = torch.load(f"/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2_layer_{layer}/diffing/thresholds.pt")
    base_active = torch.where(base_threshold > 0, 1, 0)
    rl_active = torch.where(rl_threshold > 0, 1, 0)

    



    out_dir = f"/home/woody/b114cb/b114cb23/boxo/dpo_noelia/diffing_layer_{layer}"
    os.makedirs(out_dir, exist_ok=True)

    # Create PDF file for this layer
    pdf_filename = f"{out_dir}/layer_{layer}_analysis.pdf"
    
    with PdfPages(pdf_filename) as pdf:

        rl_active = rl_active * base_active

        print(base_threshold)

        base_sae = torch.load(base_path)["model_state_dict"]
        rl_sae = torch.load(rl_path)["model_state_dict"]

        W_enc_base = base_sae["W_enc"]
        W_enc_rl = rl_sae["W_enc"]

        W_dec_base = base_sae["W_dec"]
        W_dec_rl = rl_sae["W_dec"]

        dec_cs = F.cosine_similarity(W_dec_base, W_dec_rl, dim=-1)
        enc_cs = F.cosine_similarity(W_enc_base, W_enc_rl, dim=0)

        enc_norm_base = torch.norm(W_enc_base, dim=0)
        enc_norm_rl = torch.norm(W_enc_rl, dim=0)

        dec_norm_base = torch.norm(W_dec_base, dim=-1)
        dec_norm_rl = torch.norm(W_dec_rl, dim=-1)

        diff_enc_norm = torch.abs(enc_norm_base - enc_norm_rl)
        diff_dec_norm = torch.abs(dec_norm_base - dec_norm_rl)


        #diff_enc_norm = diff_enc_norm[rl_active == 1]
        #diff_dec_norm = diff_dec_norm[rl_active == 1]
        #dec_cs = dec_cs[rl_active == 1]
        #enc_cs = enc_cs[rl_active == 1]

        # Plot 1: Cosine Similarity in Decoder (histogram)
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.hist(dec_cs.cpu().numpy(), color='skyblue')
        ax.set_title("Cosine Similarity in Decoder (base - RL)")
        ax.set_xlabel("Decoder Dimension")
        ax.set_ylabel("Cosine Similarity")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 2: Threshold Differences
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        abs_threshold_differ = torch.abs(base_threshold - rl_threshold)
        ax.bar(range(len(abs_threshold_differ)), abs_threshold_differ.cpu().numpy(), color='skyblue')
        ax.set_title("Difference in Encoder thresholds (|base - RL|)")
        ax.set_xlabel("Encoder Dimension")
        ax.set_ylabel("Threshold Difference")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 3: Norm Differences (Encoder and Decoder)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].bar(range(len(diff_enc_norm)), diff_enc_norm.cpu().numpy(), color='skyblue')
        axs[0].set_title("Difference in Encoder Norms (|base - RL|)")
        axs[0].set_xlabel("Encoder Dimension")
        axs[0].set_ylabel("Norm Difference")

        axs[1].bar(range(len(diff_dec_norm)), diff_dec_norm.cpu().numpy(), color='salmon')
        axs[1].set_title("Difference in Decoder Norms (|base - RL|)")
        axs[1].set_xlabel("Decoder Dimension")
        axs[1].set_ylabel("Norm Difference")

        fig.suptitle("Norm Differences Between Base and RL Models for Encoder and Decoder", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Plot 4: Cosine Similarity (Encoder and Decoder)
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        axs[0].bar(range(len(dec_cs)), 1-dec_cs.cpu().numpy(), color='skyblue', alpha=1.0)
        axs[0].set_title("Cosine Similarity in Decoder (base - RL)")
        axs[0].set_xlabel("Decoder Dimension")
        axs[0].set_ylabel("Cosine Similarity")

        axs[1].bar(range(len(enc_cs)), 1-enc_cs.cpu().numpy(), color='salmon', alpha=1.0)
        axs[1].set_title("Cosine Similarity in Encoder (base - RL)")
        axs[1].set_xlabel("Encoder Dimension")
        axs[1].set_ylabel("Cosine Similarity")

        fig.suptitle("Cosine Similarity Between Base and RL Models for Encoder and Decoder", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # Create a DataFrame to store the metrics for each decoder direction
        has_changed = torch.isnan(base_threshold) & (rl_threshold > 0)
        metrics_df = pd.DataFrame({
            'index': range(len(dec_cs)),
            'cosine_dissimilarity': (1 - dec_cs.cpu().numpy()),
            'norm_difference': diff_dec_norm.cpu().numpy(),
            'threshold_difference': torch.abs(base_threshold - rl_threshold).cpu().numpy(),
            'has_changed': has_changed.cpu().numpy()
        })

        # Get top 10 indices by each metric
        top_by_cosine = metrics_df.sort_values('cosine_dissimilarity', ascending=False).head(10)
        top_by_norm = metrics_df.sort_values('norm_difference', ascending=False).head(10)
        top_by_threshold = metrics_df.sort_values('threshold_difference', ascending=False).head(10)

        # Create a summary table
        print("\n=== Top 10 Decoder Directions by Different Metrics ===")
        print("\nTop 10 by Cosine Dissimilarity (1 - cosine similarity):")
        print(top_by_cosine[['index', 'cosine_dissimilarity']].to_string(index=False))

        print("\nTop 10 by Norm Difference:")
        print(top_by_norm[['index', 'norm_difference']].to_string(index=False))

        print("\nTop 10 by Threshold Difference:")
        print(top_by_threshold[['index', 'threshold_difference']].to_string(index=False))

        # Save the full metrics to CSV for further analysis
        metrics_df.to_csv(f"{out_dir}/decoder_metrics_layer_{layer}.csv", index=False)

        # Plot 5: Combined scatter plot
        fig = plt.figure(figsize=(12, 8))
        plt.scatter(metrics_df['cosine_dissimilarity'], metrics_df['norm_difference'], 
                c=metrics_df['threshold_difference'], cmap='viridis', alpha=0.9)
        plt.colorbar(label='Threshold Difference')
        plt.xlabel('Cosine Dissimilarity')
        plt.ylabel('Norm Difference')
        plt.title('Relationship Between Different Metrics for Decoder Directions')

        # Annotate top 5 points by combined ranking
        combined_rank = metrics_df['cosine_dissimilarity'].rank(ascending=False) + \
                        metrics_df['norm_difference'].rank(ascending=False) + \
                        metrics_df['threshold_difference'].rank(ascending=False)
        top_combined = metrics_df.assign(combined_rank=combined_rank).sort_values('combined_rank', ascending=True).head(5)

        for _, row in top_combined.iterrows():
            plt.annotate(f"idx: {int(row['index'])}", 
                        (row['cosine_dissimilarity'], row['norm_difference']),
                        xytext=(5, 5), textcoords='offset points')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        print("\nTop 5 by Combined Ranking:")
        print(top_combined[['index', 'cosine_dissimilarity', 'norm_difference', 'threshold_difference']].to_string(index=False))

    print(f"All plots for layer {layer} saved to: {pdf_filename}")

