#!/bin/bash
# Requested resources
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=5:00:00
#SBATCH --job-name=perturb
#SBATCH --account=def-wanglab-ab
# Use this command to run the same job interactively
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sbatch "./scripts/slurm/perturb.sh" "./conf/multinews/primera/eval.yml" \
#   "./output/results/multinews/primera/peturbations/random/addition/0.1" \
#   "addition" \
#   "random" \
#   "0.1"

### Usage notes ###
# The amount of time needed will depend on the model and dataset, but it should be ~5 hours or less.
 
### Environment ###
# Add your W&B key here to enable W&B reporting (or login with wandb login)
# export WANDB_API_KEY=""

module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/7.0.0
PROJECT_NAME="open-mds"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/def-gbader/$USER/$PROJECT_NAME" || exit

### Script arguments ###
# Required arguments
CONFIG_FILEPATH="$1"           # The path on disk to the yml config file
OUTPUT_DIR="$2"                # The path on disk to save the output to
# Optional arguments
PERTURBATION=${3:-null}        # The perturbation to run
SELECTION_STRATEGY=${4:-null}  # The selection strategy to use for perturbed documents
PERTURBED_FRAC=${5:-null}      # The fraction of input documents to perturb

### Job ###
# This calls a modified version of the example summarization script from HF (with Trainer). For details,
# see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#with-trainer
WANDB_MODE=offline \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "./scripts/run_summarization.py" "./conf/base.yml" "$CONFIG_FILEPATH" \
    output_dir="$OUTPUT_DIR" \
    perturbation="$PERTURBATION" \
    selection_strategy="$SELECTION_STRATEGY" \
    perturbed_frac="$PERTURBED_FRAC" \
    perturbed_seed="42"

exit