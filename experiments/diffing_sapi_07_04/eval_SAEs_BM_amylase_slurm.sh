#!/bin/bash -l
#SBATCH --job-name=Eval_SAEs_Diffing_SAPI_BM          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80
#SBATCH --time=24:00:00                   # Time limit
#SBATCH --output=eval_SAEs_diffing_sapi_bm_%j.out             # Output file
#SBATCH --error=eval_SAEs_diffing_sapi_bm_%j.err              # Error file

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

# Load required modules
module load python
module load cuda/11.8.0
module load cudnn/8.9.6.50-11.x
source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate
# Define the base directory for output files
output_dir="experiments/diffing_sapi_07_04/configs_eval"
mkdir -p $output_dir


# Define the array of iteration identifiers or indices
iterations=($(seq 0 30))
echo "Starting the script..."

# Iterate over each identifier to create a configuration file
for i in "${iterations[@]}"; do
  echo "Processing iteration $i..."
  output_file="$output_dir/config_${i}_bm_amylase.yaml"

  cat <<EOL > $output_file
  paths:
    model_path: /home/woody/b114cb/b114cb23/models/ZymCTRL/
    sae_path: /home/woody/b114cb/b114cb23/ZymCTRLSAEs/checkpoints/diffing_sapi_07_04/M0_D${i}/diffing/
    test_set_path: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/diffing_datasets/dataset_iteration${i}/eval/
    is_tokenized: true
    out_dir: /home/woody/b114cb/b114cb23/boxo/diffing_sapi_07_04/eval_SAEs_BM_${i}_amylase/

  samples: 1000

EOL

  echo "YAML file '$output_file' generated successfully."
  python3 -m src.inference.sae_eval --config_path $output_file --percentile 50
done
