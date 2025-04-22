#!/bin/bash -l
#SBATCH --job-name=Diffing_SAPI_RL          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=diffing_sapi_rl_%j.out             # Output file
#SBATCH --error=diffing_sapi_rl_%j.err              # Error file

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


# Define the base directory for output files

python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_rl_1.yaml
python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_rl_2.yaml
python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_rl_3.yaml
python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_rl_4.yaml
python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_rl_5.yaml
