import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Monte Carlo parameters
SAMPLE_N = 1500      # how many replicate draws
SAMPLE_SIZE = 2    # how many activities per draw

def join_ablation_csvs(base_path: str) -> pd.DataFrame:
    """
    Traverse each subfolder under `base_path`, read all CSVs and
    concat into a single DataFrame with added 'model_it' and 'dir_it' cols.
    """
    frames = []
    print(base_path)
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        print(folder_path)
        if not os.path.isdir(folder_path):
            continue
        

        # folder name is like "M1_D1_pos" or "M4_D4_neg"
        model_it = folder.split("_")[0]   # e.g. "M1"
        dir_it   = folder.split("_")[-1]  # "pos" or "neg"

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".csv"):
                continue
            print(fname)
            df = pd.read_csv(os.path.join(folder_path, fname))
            print(df)
            df["model_it"] = model_it
            df["dir_it"]   = dir_it
            frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_original_distributions(joined_df_dir: str) -> pd.DataFrame:
    """
    Load the 'original' CSVs from joined_dataframes/, extract iteration & prediction,
    and return a DataFrame with columns ['model_it','prediction'].
    """
    frames = []
    for fname in sorted(os.listdir(joined_df_dir)):
        if not fname.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(joined_df_dir, fname))
        # infer iteration from 'label' field
        lab = df["label"].iloc[0]
        # e.g. label = "..._iteration15" → iter = 15, then model_it="M3" if 15/5=3
        it = int(lab.split("_")[-1].replace("iteration", "")) // 5
        model_it = f"M{it}"
        tmp = pd.DataFrame({
            "model_it": model_it,
            "prediction": df["prediction"].values
        })
        frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["model_it", "prediction"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # 1) read ablation predictions
    BASE_AB = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/steering/"
    ab_df = join_ablation_csvs(BASE_AB)
    # --- Filter for steering model ---
    # Keep only the rows corresponding to the 'steering' model_it
    ab_df = ab_df[ab_df["model_it"] == "steering"].copy()
    if ab_df.empty:
        print("No data found for model_it='steering' in ablation results.")
        exit()
    # --- End Filter ---
    # Optional: Save the filtered intermediate data
    # ab_df.to_csv("all_steering_preds_filtered.csv", index=False)

    # 2) read original distributions (using M0 as the baseline reference)
    BASE_ORIG = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/joined_dataframes"
    orig_df = load_original_distributions(BASE_ORIG)
    # We assume M0 is the relevant baseline distribution to compare against 'steering' results
    orig_df_m0 = orig_df[orig_df["model_it"] == "M0"].copy()
    if orig_df_m0.empty:
        print("Warning: No original distribution data found for M0.")
        # Create an empty DataFrame with expected columns if M0 is missing
        orig_df_m0 = pd.DataFrame(columns=["model_it", "prediction"])


    # 3) precompute MC sample‐means for each model (now just M0)
    mc_frames = []
    #for model in orig_df["model_it"].unique(): # Now only "M0"
    model = "steering" # Explicitly set model
    vals = orig_df.loc[orig_df["model_it"] == model, "prediction"].values
    if len(vals) < SAMPLE_SIZE:
        print(f"Warning: Not enough original data points for steering ({len(vals)}) to sample {SAMPLE_SIZE}. Skipping MC.")
        mc_df = pd.DataFrame(columns=["model_it", "mean_value"]) # Empty df
    else:
        for _ in range(SAMPLE_N):
            draw = np.random.choice(vals, size=SAMPLE_SIZE, replace=False)
            mc_frames.append({"model_it": model, "mean_value": draw.mean()})
        mc_df = pd.DataFrame(mc_frames)

    # 4) Compute the per‐file mean of prediction1 & prediction2
    df_mean = (
        ab_df
        .groupby(["model_it","dir_it","name"], as_index=False)
        [["prediction1","prediction2"]]
        .mean()
    )

    # 5) Melt into long form for seaborn
    df_long = df_mean.melt(
        id_vars=["model_it","dir_it"],
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )


    # 6) Plot: one violin per pred_type, side‐by‐side, faceted by model & direction
    #    UPDATED: No model facetting, add stripplot
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        x="dir_it",        # pos / neg
        y="pred_value",    # the mean prediction
        hue="pred_type",   # prediction1 vs prediction2
        # col="model_it",    # REMOVED: Only M0
        kind="violin",
        split=False,       # side‐by‐sids
        inner="quartile",
        height=5, aspect=1, # Adjusted size
        palette="Set2"
    )

    # --- Add stripplot for main data ---
    sns.stripplot(
        data=df_long,
        x="dir_it",
        y="pred_value",
        hue="pred_type",
        dodge=True, # Separate points based on hue
        jitter=0.1,
        size=2.5,
        color="black", # Changed from using hue palette
        alpha=0.5,
        ax=g.ax, # Plot on the catplot's axes
        legend=False # Avoid duplicate legend entries
    )
    # --- End stripplot ---

    g.set_axis_labels("Direction", "Mean Prediction")
    # g.set_titles("Model = {col_name}") # REMOVED: Only M0
    g.fig.suptitle("Model M0: Mean Predictions (Split by Type)") # Add overall title
    g.add_legend(title="Type")

    # overlay MC sample‐means at x=2
    # for ax in g.axes.flatten(): # Only one axis now
    ax = g.ax # Use the single axis directly
    # model = ax.get_title().split("=")[-1].strip() # No title to parse
    model = "M0" # We know the model
    samp = mc_df.loc[mc_df["model_it"] == model, "mean_value"]
    if not samp.empty:
        sns.violinplot(
            x=np.full(len(samp), 2), # Plot at position 2
            y=samp,
            ax=ax,
            color="lightgray",
            inner="quartile",
            width=0.6
        )
        # --- Add stripplot for MC data ---
        sns.stripplot(
            x=np.full(len(samp), 2), # Plot at position 2
            y=samp,
            ax=ax,
            jitter=0.1,
            size=2.5,
            color="black", # Changed from dimgray
            alpha=0.5
        )
        # --- End stripplot ---

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["pos", "neg", "original"])
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
    plt.savefig("/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/steering/steering_split_with_mc_original_M0.png") # Updated filename

    # 7) Add a 'pred_avg' column and plot that
    #    UPDATED: No model facetting, add stripplot
    df_mean["pred_avg"] = df_mean[["prediction1","prediction2"]].mean(axis=1)
    h = sns.catplot(
        data=df_mean,
        x="dir_it",
        y="pred_avg",
        # col="model_it", # REMOVED: Only M0
        kind="violin",
        inner="quartile",
        height=5, aspect=1, # Adjusted size
        color="lightblue"
    )

    # --- Add stripplot for main data ---
    sns.stripplot(
        data=df_mean,
        x="dir_it",
        y="pred_avg",
        jitter=0.1,
        size=2.5,
        color="black", # Explicitly set to black
        alpha=0.5,
        ax=h.ax # Plot on the catplot's axes
    )
    # --- End stripplot ---

    h.set_axis_labels("Direction", "Avg. Prediction")
    # h.set_titles("Model = {col_name}") # REMOVED: Only M0
    h.fig.suptitle("Model M0: Average Predictions") # Add overall title

    # for ax in h.axes.flatten(): # Only one axis now
    ax = h.ax # Use the single axis directly
    # model = ax.get_title().split("=")[-1].strip() # No title to parse
    model = "M0" # We know the model
    samp = mc_df.loc[mc_df["model_it"] == model, "mean_value"]
    if not samp.empty:
        sns.violinplot(
            x=np.full(len(samp), 2), # Plot at position 2
            y=samp,
            ax=ax,
            color="lightgray",
            inner="quartile",
            width=0.6
        )
        # --- Add stripplot for MC data ---
        sns.stripplot(
            x=np.full(len(samp), 2), # Plot at position 2
            y=samp,
            ax=ax,
            jitter=0.1,
            size=2.5,
            color="black", # Changed from dimgray
            alpha=0.5
        )
        # --- End stripplot ---
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["pos", "neg", "original"])
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for suptitle
    plt.savefig("/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/steering/steering_avg_with_mc_original_M0.png") # Updated filename

    print("Wrote plots → steering_split_with_mc_original_M0.png, steering_avg_with_mc_original_M0.png") # Updated print

    # --- Start Plot 3: Split violins per ablation for M0 ---
    # 1) Load the joined ablation‐predictions CSV (already loaded and filtered to M0)
    #    df = pd.read_csv("all_steering_preds.csv") # Use filtered ab_df
    df = ab_df # Use the already filtered DataFrame

    # 2) Melt to long form so that prediction1/2 become a single column
    df_long = df.melt(
        id_vars=["model_it","dir_it","name","index"], # model_it is always M0 now
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )

    # 3) Plot 1: split violins of pred1 vs pred2 *for each ablation*:
    #    – facet by dir_it (rows)
    #    – x-axis is the ablation name
    #    – hue is pred_type, split=True
    #    UPDATED: No model facetting, add stripplot
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        x="name",
        y="pred_value",
        hue="pred_type",
        # col="model_it", # REMOVED: Only M0
        row="dir_it",   # Keep row facetting for pos/neg
        kind="violin",
        split=True,
        inner="quartile",
        height=4, aspect=2, # Adjusted size
        sharey=False,
        palette="Set2"
    )

    # --- Add stripplot ---
    # Iterate through axes (one for pos, one for neg)
    for ax in g.axes.flatten():
        dir_it = ax.get_title().strip() # Get 'pos' or 'neg' from title
        subset_data = df_long[df_long["dir_it"] == dir_it]
        sns.stripplot(
            data=subset_data,
            x="name",
            y="pred_value",
            hue="pred_type",
            dodge=True, # Match split=True in violinplot
            jitter=0.1,
            size=2,
            color="black", # Changed from using hue palette
            alpha=0.5,
            ax=ax,
            legend=False # Avoid duplicate legend entries
        )
    # --- End stripplot ---


    # Tidy up axis labels & titles
    g.set_axis_labels("Ablation (name)", "Prediction value")
    # g.set_titles("{row_name} – Model {col_name}") # Update title format
    g.set_titles("{row_name}") # Only show row name (pos/neg)
    g.fig.suptitle("Model M0: Split Predictions per Ablation", y=1.02) # Add overall title

    # 1) Move the legend outside
    # ... existing code ...

    # 2) Overlay the base/original distribution per model (M0)
    #    at the far right of each facet
    #    and relabel xticks to split on '_' and append 'base'
    all_names = sorted(list(df["name"].unique())) # Get names from filtered df
    short_names = [n.split("_")[-1] for n in all_names]
    for ax in g.axes.flatten():
        # extract model like "M1" or "M4" - Not needed, always M0
        # title = ax.get_title()                 # e.g. "pos – Model M1"
        # model_it = title.split("Model")[-1].strip()
        model_it = "steering"

        # raw base predictions for this model
        vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
        if not vals.empty:
            sns.violinplot(
                x=[len(all_names)] * len(vals), # Plot at position len(all_names)
                y=vals,
                ax=ax,
                color="lightgray",
                inner="quartile",
                width=0.6
            )
            # --- Add stripplot for base data ---
            sns.stripplot(
                x=[len(all_names)] * len(vals), # Plot at position len(all_names)
                y=vals,
                ax=ax,
                jitter=0.1,
                size=2,
                color="black", # Changed from dimgray
                alpha=0.5
            )
            # --- End stripplot ---

        # rebuild xticks: original names + 'base'
        ticks = list(range(len(all_names) + 1))
        labels = short_names + ["base"]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90)

    plt.tight_layout(rect=[0, 0, 0.8, 0.98]) # Adjust layout for suptitle and legend
    plt.savefig("/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/steering/steering_split_violins_with_base_M0.png") # Updated filename

    # --- Start Plot 4 (Modified): Average violins per ablation for 'steering' direction only ---
    print("Generating plot: Average predictions per ablation for 'steering' direction vs M0 baseline...")

    # 1) Filter ablation data further to only include 'steering' direction
    #    and compute per-index average prediction
    ab_df_steering = ab_df[ab_df["dir_it"] == "steering"].copy()
    if ab_df_steering.empty:
        print("No data found for dir_it='steering'. Cannot generate plot.")
        exit()
    ab_df_steering["pred_avg"] = (ab_df_steering["prediction1"] + ab_df_steering["prediction2"]) / 2

    # 2) Generate MC sample‐means for the baseline M0 distribution
    mc_frames_avg = []
    model_orig = "M0" # Baseline model identifier
    vals_orig = orig_df_m0["prediction"].values # Use pre-filtered M0 data

    if len(vals_orig) < SAMPLE_SIZE:
        print(f"Warning: Not enough original M0 data points ({len(vals_orig)}) to sample {SAMPLE_SIZE}. Skipping MC baseline.")
        mc_df_avg = pd.DataFrame(columns=["model_it", "dir_it", "name", "pred_avg"]) # Empty df
    else:
        for _ in range(SAMPLE_N):
            draw = np.random.choice(vals_orig, size=SAMPLE_SIZE, replace=False)
            # Add MC results, matching 'steering' identifiers for plotting consistency
            mc_frames_avg.append({
                "model_it": "steering", # Match ablation model_it
                "dir_it":   "steering", # Match ablation dir_it
                "name":     "base",     # Special name for baseline
                "pred_avg": draw.mean()
            })
        mc_df_avg = pd.DataFrame(mc_frames_avg)

    # 3) Concat steering ablations + baseline MC results
    plot_df = pd.concat([
        ab_df_steering[["model_it","dir_it","name","pred_avg"]],
        mc_df_avg
    ], ignore_index=True)

    # 4) Determine plotting order (full names!) and palette
    abl_names = sorted(n for n in plot_df["name"].unique() if n != "base")
    order     = ["base"] + abl_names
    pal = ["lightgray"] + list(sns.color_palette("Set2", len(abl_names)))

    # 5) Draw the violin grid (now just one plot, no row facetting)
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(max(6, len(order) * 0.8), 5)) # Adjust figure size based on number of ablations
    ax = sns.violinplot(
        data=plot_df,
        x="name", y="pred_avg",
        hue="name",
        order=order,
        hue_order=order,
        palette=pal,
        inner="quartile",
        legend=False # Legend is redundant with x-axis labels and colors
    )

    # 6) Add stripplot overlay
    sns.stripplot(
        data=plot_df,
        x="name",
        y="pred_avg",
        # hue="name", # No longer need hue for color
        order=order, # Ensure points align with violins
        # hue_order=order, # No longer need hue
        # palette=pal, # Use black instead of palette
        color="black", # Set color to black
        dodge=False, # Points overlay violins directly
        jitter=0.15, # Increase jitter slightly for better visibility
        size=2.5,    # Increase point size slightly
        alpha=0.5,
        ax=ax,
        legend=False # Avoid duplicate legend entries
    )

    # 7) Tidy up labels and titles
    ax.set_xticks(range(len(order)))
    labels = ["base (M0)"] + [n.split("_")[-1] for n in abl_names] # Clarify base is M0
    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=9) # Rotate more, adjust alignment
    ax.set_xlabel("Ablation / Baseline")
    ax.set_ylabel("Avg. Prediction")
    plt.title("Steering Ablations: Average Predictions vs. M0 Baseline", y=1.03) # Adjusted title

    plt.tight_layout()
    new_filename = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/steering/steering_avg_ablation_vs_base_M0.png"
    plt.savefig(new_filename, dpi=150)
    # plt.show() # Keep commented out unless interactive display is needed

    print(f"Finished plotting. Saved: {new_filename}")
