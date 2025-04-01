#!/bin/bash -l






# Number of top k features to use
top_k=32
# Dimension of SAE
d_sae=15360
# Layer
layer=5
# Current date
date=$(date +%Y_%m_%d)
output_dir="configs/sae_training_2b_${top_k}/"
mkdir -p $output_dir
# Define the array of iteration identifiers or indices
iterations=("0" "1" "2")  # Adjust as needed
cp configs/base_config_alex_new.yaml $output_dir/base_config_alex_new.yaml
# Iterate over each identifier to create a configuration file and submit job
for i in "${iterations[@]}"; do
  output_file="$output_dir/config_sae_2b_iter_${i}.yaml"
  
  # Determine resume settings based on iteration
  if [ "$i" = "0" ]; then
    checkpoint_dir="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_${date}_${top_k}_${d_sae}/sae_training_iter_0/"
    resume_from=""
    resuming="false"
  else
    checkpoint_dir="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_${date}_${top_k}_${d_sae}/sae_training_iter_${prev_iter}/final"
    resume_from=""
  fi
  
  # Calculate tokens for this iteration (increase with each iteration)
  
  # Create config file for this iteration
  cat <<EOL > $output_file

base_config: base_config_alex_new.yaml
base:
  model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
  d_sae: ${d_sae}
  batch_size: 4096
  model_batch_size: 512 
  seq_len: 512
sae:
  model_type: BatchTopKSAE
  sae_type: BathcTopKSAE
  site: resid_pre
  layer: ${layer}

training:
  checkpoint_dir: ${checkpoint_dir}
  lr: 0.0005
  num_tokens: 200000000
  name: "sae_training_iter_${i}"
  threshold_compute_freq: 100
  threshold_num_batches: 20 
  num_batches_in_buffer: 5
  perf_log_freq: 100
  checkpoint_freq: 1000
  top_k: ${top_k}  
  top_k_aux: 512 
  aux_penalty: 0.031
  datset_path: /home/woody/b114cb/b114cb23/boxo/new_dataset_concat_train/
  
resuming:
  resume_from: ${resume_from}
  resuming: ${resuming}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/SAE_${date}_${top_k}_${d_sae}/sae_training_iter_${i}/
EOL

  echo "YAML file '$output_file' generated successfully."
  
  # Create Slurm job script for this iteration
  slurm_script="$output_dir/slurm_sae_2b_iter_${i}.sh"
  
  cat <<EOL > $slurm_script
#!/bin/bash -l
#SBATCH --job-name=SAE_Train_2b_${i}         # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_sae_${i}_%j.out    # Output file
#SBATCH --error=slurm_sae_${i}_%j.err     # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate

# Run the Python script with the specific config
python3 -m run_topk --config ${output_file}
EOL

  chmod +x $slurm_script
  echo "Slurm script '$slurm_script' generated successfully."
  
  # Submit job and capture job ID
  if [ "$i" = "0" ]; then
    # First job doesn't need dependency
    job_id=$(sbatch $slurm_script | awk '{print $4}')
    echo "Submitted first job with ID: $job_id"
  else
    # Subsequent jobs depend on previous job completion
    prev_job_id=$job_id
    job_id=$(sbatch --dependency=afterok:$prev_job_id $slurm_script | awk '{print $4}')
    echo "Submitted job $i with ID: $job_id, dependent on job: $prev_job_id"
  fi
done

echo "All SAE training jobs have been submitted to the queue."
