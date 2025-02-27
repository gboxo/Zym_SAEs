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
source activate /home/woody/b114cb/b114cb23/boxo/pSAE/


# Run the Python script and redirect output to a log file
python3 -m run_topk.py  --is_alex True --config configs/alex.yaml 




python3 -m compute_threshold.py --sae_path /home/woody/b114cb/b114cb23/boxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_90000 --model_path /home/woody/b114cb/b114cb23/models/ZymCTRL/ --n_batches 1000


python3 -m sae_eval.py --sae_path /home/woody/b114cb/b114cb23/boxo/ZymCTRL/checkpoints/ZymCTRL_25_02_25_h100_blocks.26.hook_resid_pre_10240_batchtopk_100_0.0003_90000 --model_path /home/woody/b114cb/b114cb23/models/ZymCTRL/ --test_set_path micro_brenda.txt --is_tokenized False