#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="configs"
mkdir -p "$output_dir"




model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
df_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/activity_predictions.csv"
output_dir="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model"
sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/sae_25/diffing/"

seq_col_id="sequence"
pred_col_id="prediction2"
col_id="index"

model_name="dms_model"
ec_label="3.2.1.1"
DMS=false
hook_point="blocks.25.hook_resid_pre"

prefix_tokens=9
percentiles="[50, 75]"
min_rest_fraction="[0.05, 0.1]"







  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_generate_dms_model.yaml"
  cat <<EOL > "$gen_cfg"

    paths:
      df_path: $df_path
      out_dir: $output_dir
      model_path: $model_path
      sae_path: $sae_path
    
    df:
      seq_col_id: $seq_col_id
      pred_col_id: $pred_col_id
      col_id: $col_id
    
    model_name: $model_name
    label: $ec_label
    is_DMS: $DMS
    hook_point: $hook_point
    prefix_tokens: $prefix_tokens
    percentiles: $percentiles
    min_rest_fraction: $min_rest_fraction

EOL

python3 -m src.tools.diffing.latent_scoring --cfg_path $gen_cfg



