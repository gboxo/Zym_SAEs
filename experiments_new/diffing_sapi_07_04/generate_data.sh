#!/bin/bash -l
#SBATCH --job-name=Generate_Data    # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%A_%a.out          # Output file (%A = job ID, %a = array index)
#SBATCH --error=slurm_%A_%a.err           # Error file
#SBATCH --array=0-9                     # Run iterations 2 through 30

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
source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate



# Generate the sequences
python3 -m src.tools.generate.seq_gen --cfg_path seq_gen_cfg.yaml --iteration_num ${SLURM_ARRAY_TASK_ID}

# Compute the sequence activities
python3 -m src.tools.oracles.activity_prediction --cfg_path activity_prediction_cfg.yaml --iteration_num ${SLURM_ARRAY_TASK_ID}

# Fold the sequences
python3 -m src.tools.oracles.ESM_Fold --cfg_path oracle_cfg.yaml --iteration_num ${SLURM_ARRAY_TASK_ID}

# Compute the TM score
label="3.2.1.1"
echo foldseek started for 1B2Y
export PATH=/home/woody/b114cb/b114cb23/foldseek/bin/:$PATH
foldseek easy-search \
    "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/outputs_fold/output_iterations$((${SLURM_ARRAY_TASK_ID}))/PDB" \
    "1B2Y.pdb" \
    "/home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/TM_scores/TM_scores_${label}_iteration$((${SLURM_ARRAY_TASK_ID}))" \
    tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" \
    --exhaustive-search 1 -e inf --tmscore-threshold 0.0


# Get the dataframe for the generated sequences
python3 -m src.tools.data_utils.create_dataframe --cfg_path generated_sequences_create_dataframe_cfg.yaml --iteration_num ${SLURM_ARRAY_TASK_ID}







