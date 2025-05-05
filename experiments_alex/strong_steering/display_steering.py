import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Monte Carlo parameters
# SAMPLE_N = 1500      # how many replicate draws
# SAMPLE_SIZE = 2    # how many activities per draw

def join_results_csvs(base_path: str) -> pd.DataFrame:
    """
    Traverse each subfolder under `base_path`, read all CSVs and
    concat into a single DataFrame with added 'model_it' and 'dir_it' cols.
    Assumes CSVs are named like 'activity_predictions_pos.csv' or 'activity_predictions_neg.csv'.
    """
    frames = []
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # folder name is like "M1_D1_pos" or "M4_D4_neg"
        model_it = folder.split("_")[0]   # e.g. "M1"
        dir_it   = folder.split("_")[-1]  # "pos" or "neg"

        # Construct expected CSV filename based on dir_it
        expected_fname = f"activity_predictions_{dir_it}.csv"
        csv_path = os.path.join(folder_path, expected_fname)

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Ensure required columns exist, might need adjustment based on actual CSV structure
            if "prediction1" not in df.columns or "prediction2" not in df.columns:
                 print(f"Warning: Skipping {csv_path}. Missing 'prediction1' or 'prediction2'.")
                 continue # Or handle differently if columns are named otherwise
            df["model_it"] = model_it
            df["dir_it"]   = dir_it
            # Add 'name' column based on filename/folder if needed for grouping later
            # Example: df['name'] = folder # Or extract from CSV if available

            # Attempt to add 'name' from filename if not present
            if 'name' not in df.columns:
                 # Try to extract feature name like 'F4_D4' from 'M4_F4_D4_pos'
                 parts = folder.split('_')
                 if len(parts) >= 3: # Assuming structure like M*_F*_D*_dir
                     feature_name = "_".join(parts[1:-1]) # Join middle parts like F4_D4
                     df['name'] = feature_name
                 else:
                     # Fallback if name extraction fails
                     print(f"Warning: Could not extract feature 'name' from folder {folder}. Using index placeholder.")
                     df['name'] = f'feature_{df.index.astype(str)}'

            frames.append(df)
        else:
            print(f"Warning: Expected CSV not found: {csv_path}")


    if not frames:
        return pd.DataFrame()
    # Need to check if 'name' column exists before concatenating if it's used later
    # For now, assuming 'name' might not be strictly needed or is handled differently
    return pd.concat(frames, ignore_index=True)


def load_original_distributions(dms_data_dir: str) -> pd.DataFrame:
    """
    Load the 'original' CSVs, extract iteration & prediction,
    and return a DataFrame with columns ['model_it','prediction'].
    """
    df = pd.read_csv(dms_data_dir)
    # Assuming 'activity_dp7' is the column for the original distribution
    # And 'model_it' needs to be defined or inferred. Here, assigning 'M1' as placeholder.
    # Adjust if your original data CSV has a model identifier.
    if 'activity_dp7' not in df.columns:
        print(f"Error: 'activity_dp7' column not found in {dms_data_dir}")
        return pd.DataFrame()

    activity_dp7 = df["activity_dp7"]
    activity_dp7_no_nan = activity_dp7.dropna()
    df_new = pd.DataFrame()
    df_new["prediction"] = activity_dp7_no_nan
    # Assign a default model identifier if not present in the source CSV
    # If your CSV has a column indicating the model, use that instead.
    df_new["model_it"] = "M1" # Or extract/map from another column if available
    return df_new



if __name__ == "__main__":
    # Define base paths
    BASE_CLIP = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/steering/steering/"
    BASE_ORIG = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
    FIG_DIR   = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/steering_figures/"
    ALL_PREDS_CSV = os.path.join(FIG_DIR, "all_steering_preds_pos_only.csv")
    os.makedirs(FIG_DIR, exist_ok=True)


    # 1) read clipping predictions
    clip_df = join_results_csvs(BASE_CLIP)

    print("Clipping DataFrame (raw):")
    print(clip_df.head())
    # clip_df.to_csv(ALL_PREDS_CSV, index=False) # Maybe save before filtering?
    # print(f"Saved all clipping predictions to {ALL_PREDS_CSV}")

    # 2) read original distributions
    orig_df = load_original_distributions(BASE_ORIG)
    if orig_df.empty:
        print("\nOriginal DataFrame is empty or could not be loaded. Exiting.")
        exit()
    print("\nOriginal DataFrame:")
    print(orig_df.head())

    # Check if clip_df is empty before proceeding
    if clip_df.empty:
        print("\nClipping DataFrame is empty. Cannot proceed.")
        exit()

    # Filter to keep only 'pos' direction
    clip_df = clip_df[clip_df['dir_it'] == 'pos'].copy()
    print("\nClipping DataFrame (filtered for 'pos' direction):")
    if 'name' not in clip_df.columns:
        print("\nError: 'name' column missing after loading/filtering clipping data.")
        exit()
    print(clip_df.head())

    # Save the filtered 'pos' predictions
    clip_df.to_csv(ALL_PREDS_CSV, index=False)
    print(f"Saved 'pos' clipping predictions to {ALL_PREDS_CSV}")

    # Check if clip_df is empty after filtering
    if clip_df.empty:
        print("\nClipping DataFrame is empty after filtering for 'pos'. Cannot proceed.")
        exit()

    # Check for required columns
    required_cols = ["model_it", "dir_it", "name", "prediction1", "prediction2"]
    if not all(col in clip_df.columns for col in required_cols):
        print(f"\nError: Clipping DataFrame missing required columns. Found: {clip_df.columns}. Required: {required_cols}")
        exit()

    # 4) Compute the per‐file mean of prediction1 & prediction2 from clipping data
    df_mean = (
        clip_df
        .groupby(["model_it","dir_it","name"], as_index=False) # dir_it will always be 'pos'
        [["prediction1","prediction2"]]
        .mean()
    )
    print("\nMean Predictions DataFrame:")
    print(df_mean)

    # 5) Melt into long form for seaborn
    df_long = df_mean.melt(
        id_vars=["model_it","dir_it","name"], # Added 'name' here
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )
    print("\nLong Form DataFrame for Plot 1:")
    print(df_long)


    # 6) Plot: violins for 'pos' mean preds, plus original distribution
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        x="dir_it",        # Will only be 'pos' (at x=0)
        y="pred_value",
        hue="pred_type",   # prediction1 vs prediction2
        col="model_it",
        kind="violin",
        split=False,
        inner="quartile",
        height=4, aspect=0.9, # Adjusted aspect
        palette="Set2",
        legend=False # Disable default legend
    )

    # Overlay individual points for 'pos' direction means
    for ax in g.axes.flatten():
        try:
            model_it = ax.get_title().split("=")[-1].strip()
            subset_data = df_long[(df_long['model_it'] == model_it)]
            sns.stripplot(
                data=subset_data, x="dir_it", y="pred_value", hue="pred_type",
                ax=ax, dodge=True, jitter=0.15, color='black', size=2.5, alpha=0.6 # Increased size slightly
            )

            # Plot original distribution violin at x=1
            orig_vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
            if not orig_vals.empty:
                sns.violinplot(
                    x=np.full(len(orig_vals), 1), # Plot at x=1
                    y=orig_vals,
                    ax=ax,
                    color="lightgray",
                    inner="quartile",
                    width=0.6
                )
                # Overlay original raw points
                sns.stripplot(
                    x=np.full(len(orig_vals), 1), # Plot points at x=1
                    y=orig_vals,
                    ax=ax,
                    jitter=0.15,
                    color='dimgray', # Different color for original points
                    size=2.5,
                    alpha=0.6
                )

            # Manually add legend for pred_type (handles first N elements)
            handles, labels = ax.get_legend_handles_labels()
            n_types = len(subset_data['pred_type'].unique())
            if handles:
                ax.legend(handles[:n_types], labels[:n_types], title="Type", loc='upper right')

        except Exception as e:
            print(f"Warning: Could not overlay elements for plot 1 axis {ax.get_title()}: {e}")

    # Adjust axes and titles
    g.set_axis_labels("Direction", "Mean Prediction") # X label broader now
    g.set_titles("Model = {col_name}")

    # Adjust ticks and labels for 'pos' and 'original'
    for ax in g.axes.flatten():
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pos", "original"])

    plt.tight_layout()
    plot1_path = os.path.join(FIG_DIR, "steering_split_points_orig_pos_only.png") # Updated filename
    plt.savefig(plot1_path)
    print(f"\nSaved plot 1 to {plot1_path}")


    # 7) Plot average prediction for 'pos' plus original distribution
    df_mean["pred_avg"] = df_mean[["prediction1","prediction2"]].mean(axis=1)
    print("\nMean Predictions DataFrame with Average:")
    print(df_mean)

    h = sns.catplot(
        data=df_mean,
        x="dir_it", # Will only be 'pos' (at x=0)
        y="pred_avg",
        col="model_it",
        kind="violin",
        inner="quartile",
        height=4, aspect=0.9, # Adjusted aspect
        color="lightblue"
    )

    # Overlay individual points for 'pos' averages and original raw data
    for ax in h.axes.flatten():
        try:
            model_it = ax.get_title().split("=")[-1].strip()
            subset_data = df_mean[(df_mean['model_it'] == model_it)]
            # Stripplot for 'pos' averages
            sns.stripplot(
                data=subset_data, x="dir_it", y="pred_avg",
                ax=ax, jitter=0.15, color='black', size=2.5, alpha=0.6
            )

            # Plot original distribution violin at x=1
            orig_vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
            if not orig_vals.empty:
                sns.violinplot(
                    x=np.full(len(orig_vals), 1), y=orig_vals, ax=ax,
                    color="lightgray", inner="quartile", width=0.6
                )
                # Overlay original raw points
                sns.stripplot(
                    x=np.full(len(orig_vals), 1), y=orig_vals, ax=ax,
                    jitter=0.15, color='dimgray', size=2.5, alpha=0.6
                )

        except Exception as e:
            print(f"Warning: Could not overlay elements for plot 2 axis {ax.get_title()}: {e}")

    h.set_axis_labels("Direction", "Avg. Prediction") # X label broader
    h.set_titles("Model = {col_name}")

    # Adjust ticks and labels for 'pos' and 'original'
    for ax in h.axes.flatten():
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pos", "original"])

    plt.tight_layout()
    plot2_path = os.path.join(FIG_DIR, "steering_avg_points_orig_pos_only.png") # Updated filename
    plt.savefig(plot2_path)
    print(f"Saved plot 2 to {plot2_path}")


    # --- Plotting individual features ---

    # 1) Use the already filtered clip_df
    df_indiv = clip_df # Contains only 'pos' direction
    print("\nDataFrame for Individual Feature Plots:")
    print(df_indiv.head())

    # 2) Melt to long form so that prediction1/2 become a single column
    id_vars_melt = ["model_it", "dir_it", "name"]
    df_long_indiv = df_indiv.melt(
        id_vars=id_vars_melt,
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )
    print("\nLong Form DataFrame for Plot 3:")
    print(df_long_indiv.head())


    # 3) Plot: split violins of pred1 vs pred2 *for each clipping feature*, plus original
    sns.set_theme(style="whitegrid")
    feature_order = sorted(df_indiv["name"].unique())
    g_split = sns.catplot(
        data=df_long_indiv,
        x="name", y="pred_value", hue="pred_type", col="model_it",
        kind="violin", split=True, inner="quartile",
        height=4, aspect=1.5, # Adjusted aspect for more space
        sharey=False, palette="Set2",
        order=feature_order, # Order for features
        legend=False
    )

    # Overlay individual points for features using map_dataframe
    try:
        g_split.map_dataframe(
            sns.stripplot,
            x="name", y="pred_value", hue="pred_type",
            order=feature_order, # Match order
            hue_order=df_long_indiv['pred_type'].unique(),
            dodge=True, jitter=0.1, palette=['black', 'dimgray'],
            marker='.', size=3, alpha=0.4
        )
    except Exception as e:
        print(f"Warning: Could not overlay feature stripplot on plot 3: {e}")

    # Add legend manually
    g_split.add_legend(title="Pred Type", loc='center right', bbox_to_anchor=(1.05, 0.5))


    # Overlay the original distribution per model at the far right
    all_names = feature_order
    short_names = [n.split("_")[-1] for n in all_names]
    base_pos = len(all_names) # Position for original distribution

    for ax in g_split.axes.flatten():
        try:
            title = ax.get_title()
            model_it = title.split("Model")[-1].split("(")[0].strip() # Careful extraction

            # raw base predictions for this model
            orig_vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
            if not orig_vals.empty:
                # Plot base violin
                sns.violinplot(
                    x=np.full(len(orig_vals), base_pos), y=orig_vals, ax=ax,
                    color="lightgray", inner="quartile", width=0.6
                )
                # Plot base points
                sns.stripplot(
                    x=np.full(len(orig_vals), base_pos), y=orig_vals, ax=ax,
                    jitter=0.1, color='dimgray', marker='.', size=3, alpha=0.4
                )

            # rebuild xticks: feature names + 'original'
            ticks = list(range(len(all_names) + 1))
            labels = short_names + ["original"] # Use short names + original
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=90, fontsize=8)

        except Exception as e:
            print(f"Warning: Could not overlay original distribution on plot 3 axis {ax.get_title()}: {e}")


    # Tidy up axis labels & titles
    g_split.set_axis_labels("Clipping Feature / Original", "Prediction value") # Updated label
    g_split.set_titles("Model {col_name} (Positive Direction)")

    plt.tight_layout(rect=[0, 0, 0.95, 1]) # Adjust rect for legend if needed
    plot3_path = os.path.join(FIG_DIR, "steering_split_violins_points_orig_pos_only.png") # Updated filename
    plt.savefig(plot3_path)
    print(f"Saved plot 3 to {plot3_path}")


    # --- Plotting Average Predictions per Feature ---

    # 1) compute per‐index average for clipping data
    df_indiv["pred_avg"] = (df_indiv["prediction1"] + df_indiv["prediction2"]) / 2

    # 2) Prepare original data for concatenation
    #    Assign 'pred_avg' (using 'prediction' column), 'name' as 'original', and match 'dir_it'
    orig_plot_df = orig_df.rename(columns={'prediction': 'pred_avg'})
    orig_plot_df['name'] = 'original'
    orig_plot_df['dir_it'] = 'pos' # Match the 'dir_it' of the main data for faceting consistency

    print("\nOriginal DataFrame prepared for Plot 4:")
    print(orig_plot_df.head())


    # 3) concat clipping averages + formatted original data
    plot_df = pd.concat([
        df_indiv[["model_it","dir_it","name","pred_avg"]],
        orig_plot_df[["model_it","dir_it","name","pred_avg"]] # Ensure columns match
    ], ignore_index=True)
    print("\nCombined DataFrame for Plot 4:")
    print(plot_df.head())


    # 4) determine ordering including 'original'
    feature_names = sorted(n for n in plot_df["name"].unique() if n != "original")
    order = ["original"] + feature_names # Put original first

    # 5) build palette list: first for 'original', then one color per feature
    pal = ["lightgray"] + list(sns.color_palette("Set2", len(feature_names)))

    # 6) draw the violin grid with explicit order & hue_order
    sns.set_theme(style="whitegrid")
    g_avg = sns.catplot(
        data=plot_df,
        x="name", y="pred_avg",
        hue="name",
        col="model_it",
        # row="dir_it", # Removed row facet
        kind="violin",
        order=order,
        hue_order=order,
        palette=pal,
        inner="quartile",
        height=4, aspect=1.5, # Adjusted aspect
        sharey=False,
        legend=False # Disable legend
    )

    # 7) Overlay individual points using map_dataframe
    #    We need to handle colors carefully: gray for 'original', black for features
    point_palette = {name: ('dimgray' if name == 'original' else 'black') for name in order}

    try:
        g_avg.map_dataframe(
            sns.stripplot,
            x="name", y="pred_avg", hue="name", # Use hue to get correct positions
            order=order, hue_order=order, # Match order
            palette=point_palette, # Use custom point colors
            jitter=0.1, marker='.', size=3, alpha=0.4,
            legend=False # Do not add points to legend
        )
    except Exception as e:
        print(f"Warning: Could not overlay stripplot on plot 4: {e}")


    # 8) relabel xticks and titles
    for ax in g_avg.axes.flatten():
        ax.set_xticks(range(len(order)))
        # Use short names for features if desired
        labels = ["original"] + [n.split("_")[-1] for n in feature_names]
        # labels = order # Use full names if preferred
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel("") # Keep x label empty or set explicitly
        ax.set_ylabel("Avg. Prediction")

    g_avg.set_titles("Model {col_name} (Positive Direction)")
    plt.tight_layout()
    plot4_path = os.path.join(FIG_DIR, "steering_avg_points_orig_fixed_pos_only.png") # Updated filename
    plt.savefig(plot4_path, dpi=150)
    print(f"Saved plot 4 to {plot4_path}")

    print("\nScript finished (displaying only 'pos' direction + original, with data points).")