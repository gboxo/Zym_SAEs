#!/bin/bash -l
#SBATCH --job-name=Generate_and_score_with_ablation          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=1:00:00                   # Time limit
#SBATCH --output=SW_MD_RL_%j.out             # Output file
#SBATCH --error=SW_MD_RL_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

set -e
set -u
set -o pipefail
bash /home/woody/b114cb/b114cb23/boxo/create_venv.sh
cd $TMPDIR
source venv/bin/activate
cd /home/hpc/b114cb/b114cb23/SAETraining/crg_boxo/


# Define the base directory for output files
output_dir="experiments_alex/noelia_dpo/configs"
mkdir -p $output_dir
cp experiments_alex/old_experiments/diffing_sapi_multi_iterations_clean/configs/base_config_alex_new.yaml $output_dir/
# Define the array of iteration identifiers or indices
# Iterations 0 to 30



# Iterate over each identifier to create a configuration file
output_file="$output_dir/config_3_rl.yaml"
resume_from="/home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/finetune_SAE_DMS/diffing/"
model_path="/home/woody/b114cb/b114cb23/DPO_amylase_run_SAPI_FT_v2/output_iteration3"
#model_path="/home/woody/b114cb/b114cb23/models/model-3.2.1.1/"
dataset_path="/home/woody/b114cb/b114cb23/boxo/dpo_noelia/joined_datasets/dataset_model_3"

  cat <<EOL > $output_file
base_config: base_config_alex_new.yaml

base:
  model_path: ${model_path}
sae:
  layer: 25

training:
  dataset_path: ${dataset_path}
  name: "Diffing_SAPI_M3_D3"
  perf_log_freq: 1
  threshold_compute_freq: 1
  aux_penalty: 0
  top_k_aux: 0
  num_tokens: 450000
  threshold_num_batches: 5
  num_batches_in_buffer: 5 
  lr: 0.0003
resuming:
  resume_from: ${resume_from}
  checkpoint_dir_to: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/dpo_noelia/M3_D3_rl2/
  resuming: true
  model_diffing: true
EOL
  python model_diffing.py --config $output_file

  echo "YAML file '$output_file' generated successfully."
done
