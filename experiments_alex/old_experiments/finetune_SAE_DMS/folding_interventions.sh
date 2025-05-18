#!/bin/bash -l
#SBATCH --job-name=Latent_Scoring_Diffing_SAPI_BM          # Job name
#SBATCH --gres=gpu:1  
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=folding_interventions_%j.out             # Output file
#SBATCH --error=folding_interventions_%j.err              # Error file



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
output_dir="experiments_alex/finetune_SAE_DMS/folding_interventions"
mkdir -p $output_dir


# ====== Folding Ablations ======

output_file="$output_dir/folding_ablation.yaml"
out_dir="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/ablation_with_all/importance/M0_D0_pos_new/PDB"
seqs_path="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/ablation_with_all/importance/M0_D0_pos_new"
cat <<EOL > $output_file
paths:
    output_dir: $out_dir
    seqs_path: $seqs_path

EOL

echo "YAML file '$output_file' generated successfully."
python3 -m experiments_alex.finetune_SAE_DMS.ESM_Fold --cfg_path $output_file

# ====== Folding Clipping ======


output_file="$output_dir/folding_clipping.yaml"
out_dir="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/clipping_with_all/importance/M0_D0_pos_new/PDB"
seqs_path="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/clipping_with_all/importance/M0_D0_pos_new"

cat <<EOL > $output_file
paths:
  output_dir: $out_dir
  seqs_path: $seqs_path
EOL

echo "YAML file '$output_file' generated successfully."
python3 -m experiments_alex.finetune_SAE_DMS.ESM_Fold --cfg_path $output_file



# ====== Folding Steering ======


output_file="$output_dir/folding_steering.yaml"
out_dir="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/steering/steering/M1_D1_pos_new/PDB"
seqs_path="/home/woody/b114cb/b114cb23/boxo/finetune_SAE_DMS/steering/steering/M1_D1_pos_new"

cat <<EOL > $output_file
paths:
  output_dir: $out_dir
  seqs_path: $seqs_path
EOL

echo "YAML file '$output_file' generated successfully."
python3 -m experiments_alex.finetune_SAE_DMS.ESM_Fold --cfg_path $output_file



