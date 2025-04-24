#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="experiments_alex/diffing_sapi_multi_iterations/configs_scoring"
mkdir -p "$output_dir"

# 4) Loop over iterations and pos/neg
iterations=(1 2 3 4)
echo "Starting latent scoring..."

for i in "${iterations[@]}"; do
  for dir in pos neg; do

    # 4a) Build all of the dynamic paths
    if [ "$i" -eq 0 ]; then
      model_path="/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    else
      model_path="/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration$((i * 5))"
    fi

    sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_from_DMS/M${i}_D${i}/diffing/"
    top_features_path="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/latent_scoring/latent_scoring_${i}_rl/important_features/important_features_${dir}_M${i}_D${i}.pkl"
    out_dir="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/ablation/importance/M${i}_D${i}_${dir}"

    # 4b) Generate config for the ablation‐sequence generator
    gen_cfg="$output_dir/config_generate_${i}_${dir}_rl.yaml"
    cat <<EOL > "$gen_cfg"
paths:
  df_path: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_from_DMS/joined_dataframes/dataframe_all_iteration${i}.csv
  model_path: $model_path
  sae_path: $sae_path
  top_features_path: $top_features_path
  out_dir: $out_dir
label: 3.2.1.1
model_iteration: $i
data_iteration: $i
EOL
    echo "→ Generated generation‐config: $gen_cfg"

    # 4c) Call the generator
    #python3 -m experiments_alex.diffing_sapi_multi_iterations.generate_with_ablation \
    #  --cfg_path "$gen_cfg"

    # 5) Now score those ablated sequences
    score_cfg="$output_dir/config_score_${i}_${dir}_rl.yaml"
    cat <<EOL > "$score_cfg"
paths:
  seqs_path: $out_dir
  output_path: $out_dir/activity_predictions_${dir}.csv
  oracle_path1: "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
  checkpoint_path1: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth"
  oracle_path2: "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
  checkpoint_path2: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth"
label: 3.2.1.1
EOL
    echo "→ Generated scoring‐config: $score_cfg"

    # 5c) Call the activity predictor
    python3 -m experiments_alex.diffing_sapi_multi_iterations.activity_prediction \
      --cfg_path "$score_cfg" \
      --batch_size 16

  done
done

echo "All done."





