#!/bin/bash -l
#SBATCH --job-name=Finetune_SAE_DMS          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=finetune_SAE_DMS_FT_ZT_%j.out             # Output file
#SBATCH --error=finetune_SAE_DMS_FT_ZT_%j.err              # Error file

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
python model_diffing.py --config experiments_alex/noelia_ft_dms/configs/config_bm_20.yaml
