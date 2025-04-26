import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Monte Carlo parameters
SAMPLE_N = 1500      # how many replicate draws
SAMPLE_SIZE = 2    # how many activities per draw

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
    Load the 'original' CSVs from joined_dataframes/, extract iteration & prediction,
    and return a DataFrame with columns ['model_it','prediction'].
    """
    df = pd.read_csv(dms_data_dir)
    activity_dp7 = df["activity_dp7"]
    activity_dp7_no_nan = activity_dp7.dropna()
    df_new = pd.DataFrame()
    df_new["prediction"] = activity_dp7_no_nan
    df_new["model_it"] = "M0"
    return df_new



if __name__ == "__main__":
    # Define base paths
    BASE_CLIP = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/clipping_with_all/importance/"
    BASE_ORIG = "/home/woody/b114cb/b114cb23/boxo/alpha-amylase-training-data.csv"
    FIG_DIR   = "/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/figures_clipping"
    ALL_PREDS_CSV = os.path.join(FIG_DIR, "all_clipping_preds.csv")

    os.makedirs(FIG_DIR, exist_ok=True)

    # 1) read clipping predictions
    clip_df = join_results_csvs(BASE_CLIP)
    print("Clipping DataFrame:")
    print(clip_df)
    clip_df.to_csv(ALL_PREDS_CSV, index=False)
    print(f"Saved all clipping predictions to {ALL_PREDS_CSV}")

    # 2) read original distributions
    orig_df = load_original_distributions(BASE_ORIG)
    print("\nOriginal DataFrame:")
    print(orig_df)

    # Check if clip_df is empty before proceeding
    if clip_df.empty:
        print("\nClipping DataFrame is empty. Cannot proceed with plotting.")
        exit()

    # Check for required columns after loading
    required_cols = ["model_it", "dir_it", "prediction1", "prediction2"]
    if not all(col in clip_df.columns for col in required_cols):
        print(f"\nError: Clipping DataFrame missing required columns. Found: {clip_df.columns}. Required: {required_cols}")
        exit()
    # Add 'name' column if it's missing and needed for grouping.
    # This depends on how join_results_csvs populates it or if it's needed.
    # If 'name' is derived from the CSV filename (e.g., feature ID), it needs adding in join_results_csvs.
    # For now, assuming 'name' might come from the CSV or isn't strictly needed for the first plots.
    # If groupby fails later, this is the place to check.
    if "name" not in clip_df.columns:
         print("\nWarning: 'name' column not found in clipping DataFrame. Grouping might be affected.")
         # Add a placeholder if necessary, e.g., clip_df['name'] = 'default_name'
         # Or ensure it's correctly added during CSV loading/joining.
         # For the first plots (6, 7), 'name' is used in groupby. Let's add a placeholder for now.
         # A better approach is to extract it properly in join_results_csvs if possible.
         clip_df['name'] = 'feature_' + clip_df.index.astype(str) # Example placeholder


    # 3) precompute MC sample‐means for each model (using original data)
    mc_frames = []
    for model in orig_df["model_it"].unique():
        vals = orig_df.loc[orig_df["model_it"] == model, "prediction"].values
        if len(vals) < SAMPLE_SIZE:
             print(f"Warning: Not enough original data points for model {model} to sample {SAMPLE_SIZE}. Skipping MC.")
             continue
        for _ in range(SAMPLE_N):
            draw = np.random.choice(vals, size=SAMPLE_SIZE, replace=False)
            mc_frames.append({"model_it": model, "mean_value": draw.mean()})
    mc_df = pd.DataFrame(mc_frames)
    print("\nMC Sample Means DataFrame:")
    print(mc_df)


    # 4) Compute the per‐file mean of prediction1 & prediction2 from clipping data
    # Ensure 'name' column exists for grouping
    if "name" not in clip_df.columns:
       print("\nError: 'name' column required for grouping is missing.")
       exit()

    df_mean = (
        clip_df
        .groupby(["model_it","dir_it","name"], as_index=False)
        [["prediction1","prediction2"]]
        .mean()
    )
    print("\nMean Predictions DataFrame:")
    print(df_mean)

    # 5) Melt into long form for seaborn
    df_long = df_mean.melt(
        id_vars=["model_it","dir_it"],
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )
    print("\nLong Form DataFrame for Plot 1:")
    print(df_long)

    # 6) Plot: one violin per pred_type, side‐by‐side, faceted by model & direction
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        x="dir_it",        # pos / neg
        y="pred_value",    # the mean prediction
        hue="pred_type",   # prediction1 vs prediction2
        col="model_it",    # M0, M1, …
        kind="violin",
        split=False,       # side‐by‐side violins
        inner="quartile",
        height=4, aspect=0.8,
        palette="Set2"
    )

    g.set_axis_labels("Direction", "Mean Prediction")
    g.set_titles("Model = {col_name}")
    g.add_legend(title="Type")

    # overlay MC sample‐means at x=2
    for ax in g.axes.flatten():
        title_text = ax.get_title()
        if "=" in title_text:
            model = title_text.split("=")[-1].strip()
            samp = mc_df.loc[mc_df["model_it"] == model, "mean_value"]
            if not samp.empty:
                sns.violinplot(
                    x=np.full(len(samp), 2),
                    y=samp,
                    ax=ax,
                    color="lightgray",
                    inner="quartile",
                    width=0.6
                )
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["pos", "neg", "original"])
        else:
             print(f"Warning: Could not parse model from title: {title_text}")


    plt.tight_layout()
    plot1_path = os.path.join(FIG_DIR, "clipping_split_with_mc_original_all.png")
    plt.savefig(plot1_path)
    print(f"\nSaved plot 1 to {plot1_path}")

    # 7) Add a 'pred_avg' column and plot that
    df_mean["pred_avg"] = df_mean[["prediction1","prediction2"]].mean(axis=1)
    print("\nMean Predictions DataFrame with Average:")
    print(df_mean)

    h = sns.catplot(
        data=df_mean,
        x="dir_it",
        y="pred_avg",
        col="model_it",
        kind="violin",
        inner="quartile",
        height=4, aspect=0.8,
        color="lightblue"
    )
    h.set_axis_labels("Direction", "Avg. Prediction")
    h.set_titles("Model = {col_name}")

    for ax in h.axes.flatten():
        title_text = ax.get_title()
        if "=" in title_text:
            model = title_text.split("=")[-1].strip()
            samp = mc_df.loc[mc_df["model_it"] == model, "mean_value"]
            if not samp.empty:
                sns.violinplot(
                    x=np.full(len(samp), 2),
                    y=samp,
                    ax=ax,
                    color="lightgray",
                    inner="quartile",
                    width=0.6
                )
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(["pos", "neg", "original"])
        else:
             print(f"Warning: Could not parse model from title: {title_text}")

    plt.tight_layout()
    plot2_path = os.path.join(FIG_DIR, "clipping_avg_with_mc_original_all.png")
    plt.savefig(plot2_path)
    print(f"Saved plot 2 to {plot2_path}")

    # --- Plotting individual features ---

    # 1) Load the joined clipping‐predictions CSV
    #    It must contain: model_it, dir_it, name, index, prediction1, prediction2
    #    Using clip_df loaded earlier
    df_indiv = clip_df
    print("\nDataFrame for Individual Feature Plots:")
    print(df_indiv.head())

    # Check required columns for individual plots
    required_cols_indiv = ["model_it", "dir_it", "name", "prediction1", "prediction2"]
    if not all(col in df_indiv.columns for col in required_cols_indiv):
        print(f"\nError: DataFrame for individual plots missing required columns. Found: {df_indiv.columns}. Required: {required_cols_indiv}")
        exit()


    # 2) Melt to long form so that prediction1/2 become a single column
    #    Need 'index' if it exists and is relevant, otherwise use existing columns
    id_vars_melt = ["model_it", "dir_it", "name"]
    # Check if 'index' column exists, add if needed
    # if 'index' in df_indiv.columns:
    #    id_vars_melt.append('index')

    df_long_indiv = df_indiv.melt(
        id_vars=id_vars_melt,
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )
    print("\nLong Form DataFrame for Plot 3:")
    print(df_long_indiv.head())

    # 3) Plot 1: split violins of pred1 vs pred2 *for each clipping feature*:
    #    – facet by model_it (cols) and dir_it (rows)
    #    – x-axis is the feature name ('name' column)
    #    – hue is pred_type, split=True
    sns.set_theme(style="whitegrid")
    g_split = sns.catplot(
        data=df_long_indiv,
        x="name",
        y="pred_value",
        hue="pred_type",
        col="model_it",
        row="dir_it",
        kind="violin",
        split=True,
        inner="quartile",
        height=3, aspect=1.5,
        sharey=False,
        palette="Set2",
        order=sorted(df_indiv["name"].unique())
    )

    # Tidy up axis labels & titles
    g_split.set_axis_labels("Clipping Feature (name)", "Prediction value")
    g_split.set_titles("{row_name} – Model {col_name}")

    # Rotate x-axis labels if they overlap
    g_split.set_xticklabels(rotation=90, ha='right', fontsize=8)

    # 1) Move the legend outside
    try:
        g_split.fig.subplots_adjust(right=0.85)
        legend = g_split._legend
        if legend:
            legend.set_bbox_to_anchor((1.02, 0.5))
            legend.set_loc("center left")
        else:
            g_split.add_legend(title="Pred Type", bbox_to_anchor=(1.02, 0.5), loc="center left")
    except Exception as e:
        print(f"Could not adjust legend: {e}")


    # 2) Overlay the base/original distribution per model (RAW values, not MC means)
    #    at the far right of each facet
    #    and relabel xticks to split on '_' and append 'base'
    all_names = sorted(list(df_indiv["name"].unique()))
    short_names = [n.split("_")[-1] for n in all_names]

    for i, ax in enumerate(g_split.axes.flatten()):
        # extract model like "M0" or "M1"
        title = ax.get_title()
        if "Model" in title:
             model_it = title.split("Model")[-1].strip()

             # raw base predictions for this model
             vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
             if not vals.empty:
                 # Plot base violin at a position beyond the last feature name
                 base_pos = len(all_names)
                 sns.violinplot(
                     x=[base_pos] * len(vals),
                     y=vals,
                     ax=ax,
                     color="lightgray",
                     inner="quartile",
                     width=0.6
                 )

             # rebuild xticks: original names + 'base'
             ticks = list(range(len(all_names) + 1))
             labels = short_names + ["base"]
             ax.set_xticks(ticks)
             ax.set_xticklabels(labels, rotation=90, fontsize=8)
        else:
            print(f"Warning: Could not parse model from title for overlay: {title}")


    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plot3_path = os.path.join(FIG_DIR, "clipping_split_violins_with_base_all.png")
    plt.savefig(plot3_path)
    print(f"Saved plot 3 to {plot3_path}")

    # --- Plotting Average Predictions per Feature ---

    # 1) compute per‐index average (using df_indiv from clipping data)
    df_indiv["pred_avg"] = (df_indiv["prediction1"] + df_indiv["prediction2"]) / 2
    print("\nDataFrame with Average Prediction for Plot 4:")
    print(df_indiv.head())

    # 2) MC sample‐means for base distro (as before, but structured for concat)
    mc_frames_avg = []
    # Filter models if needed (e.g., exclude M0, M5 as in original code)
    models_to_plot = [m for m in orig_df["model_it"].unique() if m not in ["M5"]]
    for model in models_to_plot:
        vals = orig_df.loc[orig_df["model_it"] == model, "prediction"].values
        if len(vals) < SAMPLE_SIZE:
             print(f"Warning: Not enough original data points for model {model} to sample {SAMPLE_SIZE} for avg plot. Skipping MC.")
             continue
        for _ in range(SAMPLE_N):
            draw = np.random.choice(vals, size=SAMPLE_SIZE, replace=False)
            # Replicate for both pos/neg facets if faceting by dir_it
            mc_frames_avg.append({
                "model_it": model,
                "dir_it":   "pos",
                "name":     "base",
                "pred_avg": draw.mean()
            })
            mc_frames_avg.append({
                "model_it": model,
                "dir_it":   "neg",
                "name":     "base",
                "pred_avg": draw.mean()
            })
    mc_df_avg = pd.DataFrame(mc_frames_avg)
    print("\nMC Sample Means DataFrame for Plot 4:")
    print(mc_df_avg)

    # 3) concat clipping averages + base MC means
    plot_df = pd.concat([
        df_indiv[["model_it","dir_it","name","pred_avg"]],
        mc_df_avg
    ], ignore_index=True)
    print("\nCombined DataFrame for Plot 4:")
    print(plot_df.head())


    # 4) determine ordering (full names!) including 'base'
    feature_names = sorted(n for n in plot_df["name"].unique() if n != "base")
    order     = ["base"] + feature_names

    # 5) build palette list: first for 'base', then one color per feature
    pal = ["lightgray"] + list(sns.color_palette("Set2", len(feature_names)))

    # 6) draw the violin grid with explicit order & hue_order
    sns.set_theme(style="whitegrid")
    g_avg = sns.catplot(
        data=plot_df,
        x="name", y="pred_avg",
        hue="name",
        col="model_it", row="dir_it",
        kind="violin",
        order=order,
        hue_order=order,
        palette=pal,
        inner="quartile",
        height=3, aspect=1.5,
        sharey=False,
        legend=False
    )

    # 7) relabel xticks to only show the last token after '_' (or keep full names)
    for ax in g_avg.axes.flatten():
        ax.set_xticks(range(len(order)))
        labels = order
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Avg. Prediction")

    g_avg.set_titles(row_template="{row_name}", col_template="Model {col_name}")
    plt.tight_layout()
    plot4_path = os.path.join(FIG_DIR, "clipping_avg_with_mc_base_fixed_all.png")
    plt.savefig(plot4_path, dpi=150)
    print(f"Saved plot 4 to {plot4_path}")

    print("\nScript finished.")