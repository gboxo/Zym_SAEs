
#!/bin/bash -l
#SBATCH --job-name=Sequence_Generation    # Job name
#SBATCH --ntasks=1                        # uuRun 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%A.out          # Output file (%A = job ID, %a = array index)
#SBATCH --error=slurm_%A.err           # Error file

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
source /home/woody/b114cb/b114cb23/boxo/pSAE/bin/activate

# SLURM_ARRAY_TASK_ID will contain the current iteration number
for i in $(seq 1 6);
do
	python3 -m experiments.Diffing_Analysis.latent_scoring --iteration_num ${i} --label 3.2.1.1 --is_rl
done
