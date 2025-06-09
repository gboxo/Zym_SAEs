#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="configs"
mkdir -p "$output_dir"



seq_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/seq_gen_3.2.1.1_dms_model.fasta"
output_path="/home/woody/b114cb/b114cb23/boxo/unified_experiments/ablation/importance/dms_model/activity_predictions.csv"





  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_activity_prediction.yaml"

  cat <<EOL > "$gen_cfg"
    paths:
        seqs_path: $seq_path
        output_path: $output_path

        oracle_path1: "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
        checkpoint_path1: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth"

        oracle_path2: "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
        checkpoint_path2: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth"

    label: 3.2.1.1



EOL

python3 -m src.tools.oracles.activity_prediction --cfg_path $gen_cfg



