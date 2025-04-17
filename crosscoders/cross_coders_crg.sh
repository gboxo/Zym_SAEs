#!/bin/bash -l
#SBATCH --job-name=Training_CrossCoders          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --mem=24000 
#SBATCH --partition=gpu
#SBATCH --time=12:00:00                   # Time limit
#SBATCH --output=training_crosscoders_%j.out             # Output file
#SBATCH --error=training_crosscoders_%j.err              # Error file

source /users/nferruz/gboxo/venv/bin/activate
set -e
set -u
set -o pipefail


echo "Training CrossCoders"
python3 train_batch_top_k.py
