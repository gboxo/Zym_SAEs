#!/bin/bash
#SBATCH --job-name=Sequence_Generation          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%j.out             # Output file
#SBATCH --error=slurm_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/


module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE/bin/activate


# Run the Python script and redirect output to a log file
for i in $(seq 2 30);
do
	python3 seq_gen.py --iteration_num ${i} --label 3.2.1.1
done
