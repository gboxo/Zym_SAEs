#!/bin/bash -l
#SBATCH --job-name=Finetune_SAE_DMS          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:7g.80gb:1
#SBATCH --mem=24000 
#SBATCH --partition=gpu
#SBATCH --time=12:00:00                   # Time limit
#SBATCH --output=finetune_SAE_DMS_%j.out             # Output file
#SBATCH --error=finetune_SAE_DMS_%j.err              # Error file

bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/
# Define the base directory for output files
python model_diffing.py --config experiments_alex/finetune_SAE_DMS/configs/config_bm.yaml
