#!/bin/bash -l
           # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/




# Define the base directory for output files
output_dir="experiments_alex/diffing_sapi_multi_iterations/configs"
mkdir -p $output_dir
cp experiments_alex/diffing_sapi_multi_iterations/configs/base_config_alex_new.yaml $output_dir/
# Define the array of iteration identifiers or indices
# Iterations 0 to 30
iterations=($(seq 0 5))
iteration_idxs=(1 6 11 16 21)



# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  output_file="$output_dir/config_${i}_rl.yaml"
  resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/finetune_SAE_DMS/iteration${iteration_idxs[i]}/diffing/"

  if [ "$i" = "0" ]; then
    model_path="/home/woody/b114cb/b114cb23/models/ZymCTRL/"
  else
    prev_iter=$((${i}-1))
    model_path="/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/output_iteration$((${i}*5))"
  fi

  cat <<EOL > $output_file
base_config: base_config_alex_new.yaml

base:
  model_path: ${model_path}
sae:
  layer: 25

training:
  dataset_path: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations_clean/dataset_iteration${i}/
  name: "Diffing_SAPI_M${i}_D${i}"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 150000
  threshold_num_batches: 5
  num_batches_in_buffer: 10
resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M${i}_D${i}_rl/
  resuming: true
  model_diffing: true
EOL
  python model_diffing.py --config $output_file

  echo "YAML file '$output_file' generated successfully."
done
