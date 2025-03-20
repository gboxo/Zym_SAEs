#!/bin/bash
#SBATCH --job-name=SAE_Diffing_BM          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=10:00:00                   # Time limit
#SBATCH --output=slurm_%j.out             # Output file
#SBATCH --error=slurm_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE/bin/activate
# Define the base directory for output files
output_dir="configs/diffing_exp3"
mkdir -p $output_dir
cp configs/base_config_alex.yaml $output_dir/


# Define the array of iteration identifiers or indices
iterations=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")  # You can add as many as you need

echo "Starting the script..."

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  echo "Processing iteration $i..."
  output_file="$output_dir/config_${i}_bm.yaml"

  if [ "$i" = "1" ]; then
    resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/sae_training_iter_0_100/final"
  else
    prev_iter=$((${i}-1))
    resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase New/M0_D${prev_iter}/diffing"
  fi

  cat <<EOL > $output_file
base_config: base_config_alex.yaml

base:
  model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
training:
  dataset_path: /home/woody/b114cb/b114cb23/boxo/diffing_datasets/dataset_iteration${i}/
  name: "Model_Diffing_M0_D${i}"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 400000
  threshold_num_batches: 5
  num_batches_in_buffer: 10
resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase New/M0_D${i}
  resuming: true
  n_iters: 20
  model_diffing: true
EOL

  echo "YAML file '$output_file' generated successfully."
  python model_diffing.py --config $output_file
done
