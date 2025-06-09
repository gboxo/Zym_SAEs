#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="configs"
mkdir -p "$output_dir"



model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
df_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/activity_predictions.csv"
output_dir="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model"
sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/sae_25/diffing/"
top_features_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/important_features/important_features_dms_model_50_0.05_ablation.pkl"
max_activations_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/features/max_activations.pkl"

seq_col_id="sequence"
pred_col_id="prediction2"
col_id="index"

ook_point="blocks.25.hook_resid_pre"






  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_activity_prediction.yaml"

  cat <<EOL > "$gen_cfg"
    paths:
        model_path: $model_path
        sae_path: $sae_path
        top_features_path: $top_features_path
        out_dir: $output_dir
        max_activations_path: $max_activations_path



    label: 3.2.1.1
    




EOL

python3 -m src.tools.generate.experiments_runner --cfg_path $gen_cfg --experiment_type ablation



