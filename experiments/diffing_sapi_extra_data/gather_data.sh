#!/bin/bash -l

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HOME=/home/woody/b114cb/b114cb23/boxo/
export WANDB_CACHE_DIR=/home/woody/b114cb/b114cb23/boxo/


source /home/woody/b114cb/b114cb23/boxo/pSAE2/bin/activate
set -e
set -u
set -o pipefail


# We convert the fasta files into a dataset (one for each iteration)
for i in $(seq 0 30);
do
    echo "Gathering data for iteration ${i}"
    python3 -m src.tools.data_utils.create_diffing_dataset --cfg_path experiments/diffing_sapi_07_04/create_diffing_dataset_cfg.yaml --iteration_num ${i}
 done

# We collect the pLDDT, activities, etc
for i in $(seq 0 29);
do
    echo "Gathering data for the dataframe"
    python3 -m src.tools.data_utils.generate_dataframe --cfg_path experiments/diffing_sapi_07_04/generate_dataframe_cfg.yaml --iteration_num ${i}
done













