#!/bin/bash -l
#SBATCH --job-name=Firing_Rates_Diffing_SAPI          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --partition=spr1tb
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=firing_rates_diffing_sapi_%j.out             # Output file
#SBATCH --error=firing_rates_diffing_sapi_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/


source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate
set -e
set -u
set -o pipefail



for i in $(seq 1 30);
do
    echo "Collecting firing rates for iteration ${i}"
    python3 -m src.tools.diffing.firing_rates --config experiments/diffing_sapi_07_04/firing_rates_cfg.yaml --model_iteration 0 --data_iteration ${i}
done













