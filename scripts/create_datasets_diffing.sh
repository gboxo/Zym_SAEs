#! /bin/bash -l

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/

set -e
set -u
set -o pipefail



for i in $(seq 0 30);
do
    echo "Creating dataset for iteration ${i}"
    python3 -m experiments.Diffing_Analysis.create_diffing_dataset --iteration_num ${i} --label 3.2.1.1
done
