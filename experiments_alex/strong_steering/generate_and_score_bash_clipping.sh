#!/bin/bash -l




# 3) Prepare the folder for YAMLs
output_dir="experiments_alex/strong_steering/configs_scoring_clipping"
mkdir -p "$output_dir"

# 4) Loop over iterations and pos/neg
echo "Starting latent scoring..."

for top_features_path in /home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base_o2/important_features/important_features_pos_M0_D0_*; do

  echo "Top features path: $top_features_path"
  if [[ "$top_features_path" == *"ablation"* ]]; then
    continue
  fi


  name=$(basename "$top_features_path" | sed 's/.*_M0_D0_\([0-9]*_[0-9.]*\).*/\1/')

  echo "Name: $name"

  model_path="/home/woody/b114cb/b114cb23/ZF_FT_alphaamylase_gerard/FT_3.2.1.1/"


  sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/strong_steering/diffing/"
  #top_features_path="/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_dms/latent_scoring_FT/important_features/important_features_${dir}_M0_D0.pkl"
  out_dir="/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base_o2/clipping_with_all/M0_D0_${name}"

  # 4b) Generate config for the clipping‐sequence generator
  gen_cfg="$output_dir/config_generate_0_${name}_rl.yaml"
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
  python3 -m experiments_alex.strong_steering.generate_with_clipping \
    --cfg_path "$gen_cfg"
  done

for top_features_path in /home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base_o2/clipping_with_all/M0_D0_*; do

  echo "Top features path: $top_features_path"
  sleep 10



  name=$(basename "$top_features_path" | sed 's/.*_M0_D0_\([0-9]*_[0-9.]*\).*/\1/')
  echo "Name: $name"


  # 4a) Build all of the dynamic paths
  model_path="/home/woody/b114cb/b114cb23/ZF_FT_alphaamylase_gerard/FT_3.2.1.1/"


  sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/strong_steering/M0_D0/diffing/"
  #top_features_path="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/latent_scoring/latent_scoring_0_new/important_features/important_features_${dir}_M0_D0.pkl"
  out_dir="/home/woody/b114cb/b114cb23/boxo/strong_steering/latent_scoring_base_o2/clipping_with_all/${name}/"


  # 5) Now score those ablated sequences
  score_cfg="$output_dir/config_score_0_${name}_rl.yaml"
  cat <<EOL > "$score_cfg"
paths:
  seqs_path: $out_dir
  output_path: $out_dir/activity_predictions.csv
  oracle_path1: "/home/woody/b114cb/b114cb23/models/esm2_t33_650M_UR50D"
  checkpoint_path1: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRa_esm2_3B/esm_GB1_finetuned.pth"
  oracle_path2: "/home/woody/b114cb/b114cb23/models/esm1v_t33_650M_UR90S_1"
  checkpoint_path2: "/home/woody/b114cb/b114cb23/Filippo/alpha_amylase_activity_predictor/LoRA_esm1v/Esm1v_GB1_finetuned.pth"
label: 3.2.1.1
EOL
  echo "→ Generated scoring‐config: $score_cfg"

  # 5c) Call the activity predictor
  python3 -m experiments_alex.strong_steering.activity_prediction \
    --cfg_path "$score_cfg" \
    --batch_size 16

done

echo "All done."


