#!/bin/bash -l
#SBATCH --job-name=Folding_with_ESMFold          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=folding_with_ESMFold_%j.out             # Output file
#SBATCH --error=folding_with_ESMFold_%j.err              # Error file

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
# Define the base directory for output files
python3 -m experiments_alex.strong_steering.ESM_Fold --config experiments_alex/strong_steering/folding_interventions/folding_base.yaml
