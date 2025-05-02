#!/bin/bash -l


# 3) Prepare the folder for YAMLs
output_dir="experiments_alex/diffing_sapi_multi_iterations_clean/configs_scoring"
mkdir -p "$output_dir"

# 4) Loop over iterations and pos/neg
echo "Starting latent scoring..."

for i in {1..4}; do
  for dir in pos neg; do

    if [ "$i" -eq 0 ]; then
      model_path="/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    else
      model_path="/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration$((i * 5))"
    fi



    sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M${i}_D${i}/diffing/"
    top_features_path="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/latent_scoring/latent_scoring_${i}/important_features/important_features_${dir}_M${i}_D${i}.pkl"
    out_dir="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/clipping_with_all/importance/M${i}_D${i}_${dir}"

    # 4b) Generate config for the clipping‐sequence generator
    gen_cfg="$output_dir/config_generate_${i}_${dir}_rl.yaml"
    cat <<EOL > "$gen_cfg"
      paths:
        model_path: $model_path
        sae_path: $sae_path
        top_features_path: $top_features_path
        out_dir: $out_dir
      label: 3.2.1.1
      model_iteration: 0
      data_iteration: 0
EOL
    echo "→ Generated generation‐config: $gen_cfg"

    # 4c) Call the generator
    python3 -m experiments_alex.diffing_sapi_multi_iterations_clean.generate_with_clipping \
      --cfg_path "$gen_cfg"
  done
done

for i in {1..4}; do
  for dir in pos neg; do

    # 4a) Build all of the dynamic paths
    if [ "$i" -eq 0 ]; then
      model_path="/home/woody/b114cb/b114cb23/models/ZymCTRL/"
    else
      model_path="/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration$((i * 5))"
    fi


    sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M${i}_D${i}/diffing/"
    top_features_path="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/latent_scoring/latent_scoring_${i}/important_features/important_features_${dir}_M${i}_D${i}.pkl"
    out_dir="/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/clipping_with_all/importance/M${i}_D${i}_${dir}"


    # 5) Now score those ablated sequences
    score_cfg="$output_dir/config_score_0_${dir}_rl.yaml"
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
    python3 -m experiments_alex.diffing_sapi_multi_iterations_clean.activity_prediction \
      --cfg_path "$score_cfg" \
      --batch_size 16

  done
done
echo "All done."





