#!/bin/bash
#SBATCH --job-name=SAE_Diffing_BM          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=1:00:00                   # Time limit
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
output_dir="configs/diffing_exp1"
mkdir -p $output_dir

# Define the array of iteration identifiers or indices
iterations=("1" "2" "3")  # You can add as many as you need

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  output_file="$output_dir/config_${i}_rl.yaml"

  if [ "$i" = "1" ]; then
    resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000"
    model_path="/home/woody/b114cb/b114cb23/models/ZymCTRL/"
  else
    prev_iter=$((${i}-1))
    resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase/M${prev_iter}_D${prev_iter}/diffing"
    model_path="/home/woody/b114cb/b114cb23/Filippo/Q4_2024/DPO/DPO_Clean/DPO_clean_alphamylase/output_iteration_${i}"
  fi

  cat <<EOL > $output_file
base_config: base_config_workstation.yaml

base:
  model_path: ${model_path}

training:
  dataset_path: /home/hpc/b114cb/b114cb23/crg_boxo/Data/Diffing/tokenized_train_dataset_iteration${i}
  name: "Model_Diffing_M${i}_D${i}"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 2000000

resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/Diffing Alpha Amylase/M${i}_D${i}
  resuming: true
  n_iters: 700
  model_diffing: true
EOL
  python model_diffing.py --config $output_file

  echo "YAML file '$output_file' generated successfully."
done
