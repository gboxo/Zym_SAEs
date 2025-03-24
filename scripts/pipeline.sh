#!/bin/bash -l
#SBATCH --job-name=GET TM Score    # Job name
#SBATCH --ntasks=1                        # uuRun 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=slurm_%A.out          # Output file (%A = job ID, %a = array index)
#SBATCH --error=slurm_%A.err           # Error file



for i in $(seq 0 20);
do
    label="3.2.1.1"
    # Calculate TM Score
    echo foldseek started for 1B2Y
    export PATH=/home/woody/b114cb/b114cb23/foldseek/bin/:$PATH
    foldseek easy-search "/home/woody/b114cb/b114cb23/boxo/outputs/output_iterations$((i-1))/PDB" "1B2Y.pdb" "/home/woody/b114cb/b114cb23/boxo/outputs/TM_scores/TM_scores_${label}_iteration$((i-1))" tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
   
done
