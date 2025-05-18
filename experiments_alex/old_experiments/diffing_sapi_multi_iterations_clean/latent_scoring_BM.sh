#!/bin/bash -l



export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/







# Define the base directory for output files
output_dir="experiments_alex/diffing_sapi_multi_iterations_clean/configs_scoring"
mkdir -p $output_dir


# Define the array of iteration identifiers or indices
iterations=($(seq 1 5))
echo "Starting the script..."

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  echo "Processing iteration $i..."
  output_file="$output_dir/config_${i}_bm.yaml"
  model_path=/home/woody/b114cb/b114cb23/models/ZymCTRL/
  sae_path=/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M0_D${i}/diffing/

  cat <<EOL > $output_file
  paths:
    model_path: $model_path
    sae_path: $sae_path
    out_dir: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/latent_scoring/latent_scoring_${i}_bm/
    cs_path: "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/cs_data/all_cs.pt"
    df_path: "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/joined_dataframes/dataframe_all_iteration{iteration_num}.csv"

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
