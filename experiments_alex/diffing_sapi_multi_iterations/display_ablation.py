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
    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)
        if not os.path.isdir(folder_path):
            continue

        # folder name is like "M1_D1_pos" or "M4_D4_neg"
        model_it = folder.split("_")[0]   # e.g. "M1"
        dir_it   = folder.split("_")[-1]  # "pos" or "neg"

        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(folder_path, fname))
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
    BASE_AB = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/ablation_with_all/importance"
    ab_df = join_ablation_csvs(BASE_AB)
    print(ab_df)
    ab_df.to_csv("all_ablation_preds.csv", index=False)

    # 2) read original distributions
    BASE_ORIG = "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/joined_dataframes"
    orig_df = load_original_distributions(BASE_ORIG)

    # 3) precompute MC sample‐means for each model
    mc_frames = []
    for model in orig_df["model_it"].unique():
        vals = orig_df.loc[orig_df["model_it"] == model, "prediction"].values
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

    print(df_long)

    # 6) Plot: one violin per pred_type, side‐by‐side, faceted by model & direction
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
        x="dir_it",        # pos / neg
        y="pred_value",    # the mean prediction
        hue="pred_type",   # prediction1 vs prediction2
        col="model_it",    # M1, M2, …
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
        model = ax.get_title().split("=")[-1].strip()
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
    plt.tight_layout()
    plt.savefig("ablation_split_with_mc_original_all.png")

    # 7) Add a 'pred_avg' column and plot that
    df_mean["pred_avg"] = df_mean[["prediction1","prediction2"]].mean(axis=1)

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
        model = ax.get_title().split("=")[-1].strip()
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
    plt.tight_layout()
    plt.savefig("ablation_avg_with_mc_original_all.png")

    print("Wrote plots → ablation_split_with_mc_original.png, ablation_avg_with_mc_original.png")

    # 1) Load the joined ablation‐predictions CSV
    #    It must contain: model_it, dir_it, name, index, prediction1, prediction2
    df = pd.read_csv("all_ablation_preds.csv")

    # 2) Melt to long form so that prediction1/2 become a single column
    df_long = df.melt(
        id_vars=["model_it","dir_it","name","index"],
        value_vars=["prediction1","prediction2"],
        var_name="pred_type",
        value_name="pred_value"
    )

    # 3) Plot 1: split violins of pred1 vs pred2 *for each ablation*:
    #    – facet by model_it (cols) and dir_it (rows)
    #    – x-axis is the ablation name
    #    – hue is pred_type, split=True
    # --- existing code: draw split violins for each ablation name ---
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df_long,
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
        palette="Set2"
    )

    # Tidy up axis labels & titles
    g.set_axis_labels("Ablation (name)", "Prediction value")
    g.set_titles("{row_name} – Model {col_name}")

    # 1) Move the legend outside
    g.fig.subplots_adjust(right=0.8)  # make room on right
    legend = g._legend
    legend.set_bbox_to_anchor((1.02, 0.5))
    legend.set_loc("center left")

    # 2) Overlay the base/original distribution per model
    #    at the far right of each facet
    #    and relabel xticks to split on '_' and append 'base'
    all_names = list(df["name"].unique())
    short_names = [n.split("_")[-1] for n in all_names]
    for ax in g.axes.flatten():
        # extract model like "M1" or "M4"
        title = ax.get_title()                 # e.g. "pos – Model M1"
        model_it = title.split("Model")[-1].strip()

        # raw base predictions for this model
        vals = orig_df.loc[orig_df["model_it"] == model_it, "prediction"]
        if not vals.empty:
            sns.violinplot(
                x=[len(all_names)] * len(vals),
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
        ax.set_xticklabels(labels, rotation=90)

    plt.tight_layout()
    plt.savefig("ablation_split_violins_with_base_all.png")

    # 1) compute per‐index average
    ab_df["pred_avg"] = (ab_df["prediction1"] + ab_df["prediction2"]) / 2

    # 2) MC sample‐means for base distro (as before)
    mc_frames = []
    for model in orig_df["model_it"].unique():
        if model == "M5" or model == "M0":
            continue
        vals = orig_df.loc[orig_df["model_it"] == model, "prediction"].values
        for _ in range(SAMPLE_N):
            draw = np.random.choice(vals, size=SAMPLE_SIZE, replace=False)
            mc_frames.append({
                "model_it": model,
                "dir_it":   "pos",    # replicate for both facets
                "name":     "base",
                "pred_avg": draw.mean()
            })
            mc_frames.append({
                "model_it": model,
                "dir_it":   "neg",
                "name":     "base",
                "pred_avg": draw.mean()
            })
    mc_df = pd.DataFrame(mc_frames)

    # 3) concat ablations + base
    plot_df = pd.concat([
        ab_df[["model_it","dir_it","name","pred_avg"]],
        mc_df
    ], ignore_index=True)

    # 4) determine ordering (full names!)
    abl_names = sorted(n for n in plot_df["name"].unique() if n != "base")
    order     = ["base"] + abl_names

    # 5) build palette list: first for 'base', then one color per ablation
    pal = ["lightgray"] + list(sns.color_palette("Set2", len(abl_names)))

    # 6) draw the violin grid with explicit order & hue_order
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
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
        legend=False   # we'll drop the legend since it'd list every ablation
    )

    # 7) relabel xticks to only show the last token after '_'
    for ax in g.axes.flatten():
        ax.set_xticks(range(len(order)))
        labels = ["base"] + [n.split("_")[-1] for n in abl_names]
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Avg. Prediction")

    g.set_titles(row_template="{row_name}", col_template="Model {col_name}")
    plt.tight_layout()
    plt.savefig("ablation_avg_with_mc_base_fixed_all.png", dpi=150)
    plt.show()