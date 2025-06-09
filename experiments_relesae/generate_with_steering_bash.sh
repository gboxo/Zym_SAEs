#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="configs"
mkdir -p "$output_dir"



model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
df_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/activity_predictions.csv"
output_dir="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model"

seq_col_id="sequence"
pred_col_id="prediction2"
col_id="index"







  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_steering.yaml"

  cat <<EOL > "$gen_cfg"
    paths:
        model_path: $model_path
        out_dir: $output_dir
        df_path: $df_path



    label: 3.2.1.1
    seq_col_id: $seq_col_id
    pred_col_id: $pred_col_id
    col_id: $col_id
    hook_point: "hook_resid_pre"
    




EOL

python3 -m src.tools.generate.experiments_runner --cfg_path $gen_cfg --experiment_type steering



