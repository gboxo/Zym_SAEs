#!/bin/bash -l
#SBATCH --job-name=Sequence_Generation    # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%A_%a.out          # Output file (%A = job ID, %a = array index)
#SBATCH --error=slurm_%A_%a.err           # Error file
#SBATCH --array=5-9                     # Run iterations 2 through 30

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

set -e
set -u
set -o pipefail

bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/



# SLURM_ARRAY_TASK_ID will contain the current iteration number
python3 -m experiments_alex.strong_steering.seq_gen 