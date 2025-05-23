#!/bin/bash -l
#SBATCH --job-name=Generate_and_score_with_ablation          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=4:00:00                   # Time limit
#SBATCH --output=Generate_and_score_with_ablation_%j.out             # Output file
#SBATCH --error=Generate_and_score_with_ablation_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

set -e
set -u
set -o pipefail
bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/

# 3) Prepare the folder for YAMLs
output_dir="experiments_alex/noelia_ft_dms/configs_scoring"
mkdir -p "$output_dir"

# 4) Loop over iterations and pos/neg
echo "Starting latent scoring..."

for top_features_path in /home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/important_features/important_features_pos_M0_D0_*; do

  echo "Top features path: $top_features_path"


  if [[ "$top_features_path" != *"ablation"* ]]; then
    echo "Not an ablation file"
    continue
  fi
  name=$(basename "$top_features_path" | sed 's/.*_M0_D0_\([0-9]*_[0-9.]*\)_ablation.*\.pkl/\1/')

  echo "Name: $name"
  model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"


  sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/diffing/"
  out_dir="/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/ablation_with_all/M0_D0_${name}"



  # 4b) Generate config for the ablation‐sequence generator
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
  python3 -m experiments_alex.noelia_ft_dms.generate_with_ablation \
    --cfg_path "$gen_cfg"
  done


for top_features_path in /home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/ablation_with_all2/M0_D0_*; do
  echo "Top features path: $top_features_path"



  name=$(basename "$top_features_path" | sed 's/.*_M0_D0_\([0-9]*_[0-9.]*\)_ablation.*/\1/')
  echo "Name: $name"


  # 4a) Build all of the dynamic paths
  model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"


  sae_path="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/diffing/"
  out_dir="/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_base/ablation_with_all2/${name}/"


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
  python3 -m experiments_alex.noelia_ft_dms.activity_prediction \
    --cfg_path "$score_cfg" \
    --batch_size 16

done

echo "All done."





