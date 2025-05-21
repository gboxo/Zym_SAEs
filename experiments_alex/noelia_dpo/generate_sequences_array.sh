#!/bin/bash -l
#SBATCH --job-name=Generate_and_score_with_ablation          # Job name
#SBATCH --ntasks=1                        # Run 1 task (process)
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --time=4:00:00                   # Time limit
#SBATCH --output=Generate_and_score_with_ablation_%j.out             # Output file
#SBATCH --error=Generate_and_score_with_ablation_%j.err              # Error file

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



# where all your model subfolders live:
SEQ_DIR="/home/woody/b114cb/b114cb23/DPO_amylase_run_SAPI_FT_v2"

# where to dump each model's generated sequences
OUTPUT_BASE="/home/woody/b114cb/b114cb23/boxo/dpo_noelia/generated_seqs_by_model"

# path to your conda/venv


#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# do not edit below this line
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# compute array size by counting model dirs
model_dirs=( $(ls -d "${SEQ_DIR}"/output_iteration*/ 2>/dev/null | sort -V) )
num_models=${#model_dirs[@]}
if [ $num_models -eq 0 ]; then
  echo "ERROR: no model directories found in ${SEQ_DIR}" >&2
  exit 1
fi

# re–submit self with proper --array if needed
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  # not yet an array job; re-sbatch with correct range
  sbatch --array=0-$((num_models-1)) "$0"
  exit
fi

# pick out the model for this task
MODEL_DIR="${model_dirs[$SLURM_ARRAY_TASK_ID]}"
echo "[$(date)] Running task ${SLURM_ARRAY_TASK_ID} on model ${MODEL_DIR}"

# now that SLURM_ARRAY_TASK_ID is guaranteed, set up the config
CFG_PATH="configs/seq_gen_iteration${SLURM_ARRAY_TASK_ID}_cfg.yaml"
mkdir -p configs

# make per‐model output
OUTPUT_DIR="${OUTPUT_BASE}/model_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT_DIR"




# Create a file named seq_gen_cfg.yaml in the cfg folder with the following content:
cat > configs/seq_gen_iteration${SLURM_ARRAY_TASK_ID}_cfg.yaml <<EOF
paths:
  model_path: "${MODEL_DIR}"
  out_dir: "${OUTPUT_DIR}"

label: 3.2.1.1
EOF



# run your generation command (adjust flags as needed)
python3 -m src.tools.generate.seq_gen \
  --cfg_path "${CFG_PATH}" \
  --iteration_num "${SLURM_ARRAY_TASK_ID}"

echo "[$(date)] Finished model ${MODEL_DIR}"
