#!/bin/bash
# Requested resources
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=5:00:00
#SBATCH --job-name=eval-checkpoint-gold
#SBATCH --account=def-wanglab-ab
# Use this command to run the same job interactively
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# Note that this script is intended be called by submit_eval_checkpoint.sh!
#
# sbatch "./scripts/slurm/eval.sh" "./conf/multinews/primera/eval.yml" \
#   "./output/multinews/primera/training/checkpoint-702" \
#   "./output/results/multinews/primera/trained_with_retrieval/checkpoint-702"

### Usage notes ###
# The amount of time needed will depend on the model and dataset, but it should be ~5 hours or less.
 
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
CONFIG_FILEPATH="$1"           # The path on disk to the yml config file
OUTPUT_DIR="$2"                # The path on disk to save the output to
MODEL_NAME_OR_PATH="$3"        # The name (or path on disk) of the model to evaluate

### Job ###
# This calls a modified version of the example summarization script from HF (with Trainer). For details,
# see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#with-trainer
WANDB_MODE=offline \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "./scripts/run_summarization.py" "./conf/base.yml" "$CONFIG_FILEPATH" \
    output_dir="$OUTPUT_DIR" \
    model_name_or_path="$MODEL_NAME_OR_PATH"

exit