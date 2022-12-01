#!/bin/bash
# Requested resources
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=9:00:00
#SBATCH --job-name=retrieve
#SBATCH --account=def-wanglab-ab
# Use this command to run the same job interactively
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sbatch "./scripts/slurm/retrieve.sh" "./conf/multinews/primera/eval.yml" \
#   "./output/results/multinews/primera/retrieval/sparse/mean" \
#   "./output/datasets/multinews_sparse_mean" \
#   "sparse" \
#   "mean"
#
# Or, for the training experiments:
# sbatch "./scripts/slurm/retrieve.sh" "./conf/multinews/primera/train_retrieved.yml" \
#   "./output/results/multinews/primera/trained_with_retrieval" \
#   "./output/datasets/multinews_dense_mean" \
#   "dense" \
#   "mean"
# 

### Usage notes ###
# Most dataset, retriever, and top-k strategy combinations should take about 5 hours or less.
# The larger datasets (e.g. Multi-News and MS2) will take longer, especially when using the max strategy.
 
### Environment ###
# Add your W&B key here to enable W&B reporting (or login with wandb login)
# export WANDB_API_KEY=""

module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/7.0.0
PROJECT_NAME="retrieval-exploration"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/def-gbader/$USER/$PROJECT_NAME" || exit

### Script arguments ###
# Required arguments
CONFIG_FILEPATH="$1"  # The path on disk to the yml config file
OUTPUT_DIR="$2"       # The path on disk to save the output to
DATASET_DIR="$3"      # The path on disk to the dataset to use
RETRIEVER="$4"        # The type of retriever to use
STRATEGY="$5"         # The strategy to use when choosing the k top documents to retrieve

### Job ###
# This calls a modified version of the example summarization script from HF (with Trainer). For details,
# see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#with-trainer

WANDB_MODE=offline \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "./scripts/run_summarization.py" "./conf/base.yml" "$CONFIG_FILEPATH" \
    output_dir="$OUTPUT_DIR" \
    dataset_name="$DATASET_DIR" \
    retriever="$RETRIEVER" \
    top_k_strategy="$STRATEGY"

exit