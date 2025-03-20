
#!/bin/bash -l
top_k=100
output_dir="configs/sae_finetune_512_ctx/"
mkdir -p $output_dir
# Define the array of iteration identifiers or indices
cp configs/base_config_alex.yaml $output_dir/base_config_alex.yaml
# Iterate over each identifier to create a configuration file and submit job
checkpoint_dir=""
resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000"
resuming="true"
output_file="$output_dir/config_sae_finetune_512_ctx.yaml"
  
# Create config file for this iteration
cat <<EOL > $output_file

base_config: base_config_alex.yaml
base:
  model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
  d_sae: 10240
  batch_size: 4096
  model_batch_size: 512 
  seq_len: 512

training:
  checkpoint_dir: ${checkpoint_dir}
  lr: 0.001
  num_tokens: 200000000
  name: "sae_finetune_512_ctx"
  threshold_compute_freq: 100
  threshold_num_batches: 20 
  num_batches_in_buffer: 5
  perf_log_freq: 100
  checkpoint_freq: 10000
  top_k: ${top_k}  
  top_k_aux: 512 
  aux_penalty: 0.01
  datset_path: /home/woody/b114cb/b114cb23/boxo/new_dataset_train/
  
resuming:
  resume_from: ${resume_from}
  resuming: ${resuming}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/sae_finetune_512_ctx
EOL

  echo "YAML file '$output_file' generated successfully."
  
  # Create Slurm job script for this iteration
  slurm_script="$output_dir/slurm_finetune_sae_512_ctx.sh"
  
  cat <<EOL > $slurm_script
#!/bin/bash
#SBATCH --job-name=SAE_Finetune_512_ctx         # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_sae_finetune_512_ctx_%j.out    # Output file
#SBATCH --error=slurm_sae_finetune_512_ctx_%j.err     # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE/bin/activate

# Run the Python script with the specific config
python3 -m run_topk --config ${output_file}
EOL

  chmod +x $slurm_script
  echo "Slurm script '$slurm_script' generated successfully."
  
  # Submit job and capture job ID
  job_id=$(sbatch $slurm_script | awk '{print $4}')
  echo "Submitted first job with ID: $job_id"


echo "All SAE training jobs have been submitted to the queue."
