#!/bin/bash -l
#SBATCH --job-name=Sequence_Folding    # Job name
#SBATCH --ntasks=1                     # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00               # Time limit
#SBATCH --output=slurm_%A_%a.out      # Output file (%A = job ID, %a = array index)
#SBATCH --error=slurm_%A_%a.err       # Error file
#SBATCH --array=0-5                   # Array range from 2 to 30

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

set -e
set -u
set -o pipefail

module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x

echo "Starting folding for iteration ${SLURM_ARRAY_TASK_ID}"
python3 -m experiments.Diffing_Analysis.ESM_Fold --iteration_num ${SLURM_ARRAY_TASK_ID} --label 3.2.1.1
