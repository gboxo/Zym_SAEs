#!/bin/bash -l
#SBATCH --job-name=SAE_Oracle          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00                   # Time limit
#SBATCH --output=sae_oracle_%j.out             # Output file
#SBATCH --error=sae_oracle_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
source /users/nferruz/gboxo/crg_boxo/pSAE/bin/activate
# Define the base directory for output files



echo "Running SAE Oracle"

python3 -m experiments.sae_oracle.main 

echo "SAE Oracle finished"


