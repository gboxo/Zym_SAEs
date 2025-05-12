#!/bin/bash -l
#SBATCH --job-name=Latent_Scoring_Diffing_DMS_noelia_Ft          # Job name
#SBATCH --gres=gpu:1  
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=4:00:00                   # Time limit
#SBATCH --output=latent_scoring_noelia_ft_dms_%j.out             # Output file
#SBATCH --error=latent_scoring_noelia_ft_dms_%j.err              # Error file



export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/



bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/



# Define the base directory for output files
output_dir="experiments_alex/noelia_ft_dms/configs_scoring"
mkdir -p $output_dir


# Define the array of iteration identifiers or indices

# Iterate over each identifier to create a configuration file
output_file="$output_dir/config_bm.yaml"
model_path=/home/woody/b114cb/b114cb23/models/model-3.2.1.1/
sae_path=/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/noelia_ft_dms/diffing/

cat <<EOL > $output_file
paths:
  model_path: $model_path
  sae_path: $sae_path
  out_dir: /home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/latent_scoring/latent_scoring_0/
  df_path: "/home/woody/b114cb/b114cb23/boxo/noelia_ft_dms/alpha-amylase-training-data.csv"

iteration_num: 0 
label: 3.2.1.1
model_iteration: 0
data_iteration: 0 
EOL

echo "YAML file '$output_file' generated successfully."
python3 -m experiments_alex.noelia_ft_dms.latent_scoring --cfg_path $output_file
