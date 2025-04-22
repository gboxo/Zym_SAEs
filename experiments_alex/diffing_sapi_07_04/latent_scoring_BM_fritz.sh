#!/bin/bash -l
#SBATCH --job-name=Latent_Scoring_Diffing_SAPI_BM          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --partition=spr1tb
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=latent_scoring_bm_%j.out             # Output file
#SBATCH --error=latent_scoring_bm_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate
# Define the base directory for output files
output_dir="experiments/diffing_sapi_07_04/configs_scoring"
mkdir -p $output_dir


# Define the array of iteration identifiers or indices
iterations=($(seq 1 30))
echo "Starting the script..."

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  echo "Processing iteration $i..."
  output_file="$output_dir/config_${i}_bm.yaml"
  if [ $i -eq 0 ]; then
    model_path=/home/woody/b114cb/b114cb23/models/ZymCTRL/
    sae_path=/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_07_04/M0_D0/diffing/
  else
    model_path=/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration${i}/
    sae_path=/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_07_04/M0_D${i}/diffing/
  fi

  cat <<EOL > $output_file
  paths:
    model_path: $model_path
    sae_path: $sae_path
    out_dir: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/latent_scoring/latent_scoring_${i}_bm/
    cs_path: "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/cs_data/all_cs.pt"
    df_path: "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/dataframes/dataframe_iteration{iteration_num}.csv"

  iteration_num: $i 
  label: 3.2.1.1
  model_iteration: 0
  data_iteration: $i 
  thresholds:
    pred: 
      upper: 2 
      lower: 1
    plddt: 
      upper: 0.7
      lower: 0.5
    tm_score: 
      upper: 0.65
      lower: 0.55

EOL

  echo "YAML file '$output_file' generated successfully."
  python3 -m src.tools.diffing.latent_scoring --cfg_path $output_file
done
