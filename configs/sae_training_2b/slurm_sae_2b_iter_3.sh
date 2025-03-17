#!/bin/bash
#SBATCH --job-name=SAE_Train_2b_3         # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_sae_3_%j.out    # Output file
#SBATCH --error=slurm_sae_3_%j.err     # Error file

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
python3 -m run_topk --config configs/sae_training_2b//config_sae_2b_iter_3.yaml
