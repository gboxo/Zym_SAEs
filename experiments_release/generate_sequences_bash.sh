#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="configs"
mkdir -p "$output_dir"


model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
out_dir="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model"


  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_generate_dms_model.yaml"
  cat <<EOL > "$gen_cfg"

    paths:
      model_path: $model_path
      out_dir: $out_dir

    model_name: "dms_model" 
    label: 3.2.1.1
    n_samples: 10
    n_batches: 10

EOL

python3 -m src.tools.generate.seq_gen --cfg_path $gen_cfg



