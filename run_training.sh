#!/bin/bash
#SBATCH --job-name=SAE_Training          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%j.out             # Output file
#SBATCH --error=slurm_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE/bin/activate



# Run the Python script and redirect output to a log file
python3 -m src.training.run_topk --config configs/alex.yaml --resume --checkpoint_path /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_200000/ --additional_iters 1_000_000

