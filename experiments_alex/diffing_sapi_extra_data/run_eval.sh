#!/bin/bash -l
#SBATCH --job-name=Eval_Diffing_SAPI          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=eval_diffing_sapi_%j.out             # Output file
#SBATCH --error=eval_diffing_sapi_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate
# Define the base directory for output files

# Evaluate the SAEs trained on the Base Model
for i in {0..25}
do
    python3 -m src.inference.sae_eval
done





# Evaluate the SAEs trained on the DPOed Model






