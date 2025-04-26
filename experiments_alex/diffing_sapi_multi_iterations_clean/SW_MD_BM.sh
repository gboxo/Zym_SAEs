#!/bin/bash -l


# Define the base directory for output files
output_dir="experiments_alex/diffing_sapi_multi_iterations/configs"
mkdir -p $output_dir
cp experiments_alex/diffing_sapi_multi_iterations/configs/base_config_alex_new.yaml $output_dir/


# Define the array of iteration identifiers or indices
iterations=($(seq 1 5))
echo "Starting the script..."

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  echo "Processing iteration $i..."
  output_file="$output_dir/config_${i}_bm.yaml"
  resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/finetune_SAE_DMS/diffing/"
  


  cat <<EOL > $output_file
base_config: base_config_alex_new.yaml

base:
  model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
sae:
  layer: 25

training:
  dataset_path: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_multi_iterations/dataset_iteration${i}/
  name: "Diffing_SAPI_M0_D${i}"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 400000
  threshold_num_batches: 5
  num_batches_in_buffer: 2
resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_multi_iterations_clean/M0_D${i}
  resuming: true
  n_iters: 20
  model_diffing: true
EOL

  echo "YAML file '$output_file' generated successfully."
  python model_diffing.py --config $output_file
done
