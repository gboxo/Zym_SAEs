#!/bin/bash -l
#SBATCH --job-name=Activity_Prediction          # Job name
#SBATCH --gres=gpu:1  
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=4:00:00                   # Time limit
#SBATCH --output=activity_prediction_%j.out             # Output file
#SBATCH --error=activity_prediction_%j.err              # Error file



export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/


# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x


bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/

python3 -m experiments_alex.oracle_ensemble.activity_prediction