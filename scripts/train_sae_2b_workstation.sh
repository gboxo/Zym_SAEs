
#!/bin/bash
# Define the base directory for output files
output_dir="configs/sae_training_2b_workstation/"
# Define the array of iteration identifiers or indices
iterations=("0" "1" "2" "3" "4")  # Adjust as needed
cp configs/base_config_workstation.yaml $output_dir/base_config_workstation.yaml

# Iterate over each identifier to create a configuration file and submit job
for i in "${iterations[@]}"; do
  output_file="$output_dir/config_sae_2b_iter_${i}.yaml"
  
  # Determine resume settings based on iteration
  if [ "$i" = "0" ]; then
    resume_from=""
    resuming="false"
  else
    prev_iter=$((${i}-1))
    resume_from="/users/nferruz/gboxo/ZymCTRL/checkpoints/sae_training_iter_${prev_iter}/final"
    resuming="true"
  fi
  
  # Calculate tokens for this iteration (increase with each iteration)
  
  # Create config file for this iteration
  cat <<EOL > $output_file

base_config: base_config_workstation.yaml
base:
  model_path: AI4PD/ZymCTRL 
  d_sae: 10240

training:
  num_tokens: 50000
  name: "sae_training_iter_${i}"
  checkpoint_dir: /users/nferruz/gboxo/ZymCTRL/checkpoints/sae_training_iter_${i}
  
resuming:
  resume_from: ${resume_from}
  resuming: ${resuming}
  checkpoint_dir_to: /users/nferruz/gboxo/ZymCTRL/checkpoints/sae_training_iter_${i}
  wandb_project: "DEBUG"

EOL

  echo "YAML file '$output_file' generated successfully."
  
  python3 -m run_topk --config ${output_file}
done
