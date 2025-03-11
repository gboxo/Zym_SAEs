#!/bin/bash

# Define the base directory for output files
output_dir="configs/diffing_exp1"
mkdir -p $output_dir

# Define the array of iteration identifiers or indices
iterations=("1" "2" "3")  # You can add as many as you need

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  output_file="$output_dir/config_${i}_rl.yaml"

  if [ "$i" = "1" ]; then
    resume_from="/users/nferruz/gboxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000"
  else
    prev_iter=$((${i}-1))
    resume_from="/users/nferruz/gboxo/ZymCTRL/checkpoints/Diffing Alpha Amylase/M${prev_iter}_D${prev_iter}/diffing"
  fi

  cat <<EOL > $output_file
base_config: base_config_workstation.yaml

base:
  model_path: /users/nferruz/gboxo/Alpha Amylase/Output Iteration ${i}/

training:
  dataset_path: /users/nferruz/gboxo/crg_boxo/Data/Diffing/tokenized_train_dataset_iteration${i}
  name: "Model_Diffing_M${i}_D${i}"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 2000000

resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /users/nferruz/gboxo/ZymCTRL/checkpoints/Diffing Alpha Amylase/M${i}_D${i}
  resuming: true
  n_iters: 700
  model_diffing: true
EOL
  python model_diffing.py --config $output_file

  echo "YAML file '$output_file' generated successfully."
done
