#!/bin/bash
# Requested resources
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=3:00:00
#SBATCH --job-name=perturb
#SBATCH --account=def-gbader
# Emails me when job starts, ends or fails
#SBATCH --mail-user=johnmgiorgi@gmail.com
#SBATCH --mail-type=FAIL
# Use this command to run the same job interactively
# salloc --mem=12G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=rrg-wanglab
# salloc --mem=16G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sh ./scripts/slurm/run.sh "./conf/multi_news/primera/eval.yml" "./output/random/addition/0.1" \
#   "addition" \
#   "random" \
#   "0.1"
 
### Environment ###
# Add your W&B key here to enable W&B reporting
# export WANDB_API_KEY=""

module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load StdEnv/2020 gcc/9.3.0 python/3.8 arrow/7.0.0
PROJECT_NAME="retrieval-exploration"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/def-gbader/$USER/$PROJECT_NAME" || exit

### Script arguments ###
# Must be provided as argument to the script
CONFIG_FILEPATH="$1"    # The path on disk to the yml config file
OUTPUT_DIR="$2"         # The path on disk to save the output to
PERTURBATION="$3"       # The perturbation to run
SELECTION_STRATEGY="$4"  # The selection strategy to use for perturbed documents
PERTURBED_FRAC="$5"     # The fraction of input documents to perturb
# Allow these to be optional
PERTURBATION=${PERTURBATION:=null}
SELECTION_STRATEGY=${SELECTION_STRATEGY:=null}
PERTURBED_FRAC=${PERTURBED_FRAC:=null}
# Constants
PERTURBED_SEED=42

### Job ###
# This calls a modified version of the example summarization script from HF (with Trainer). For details,
# see: https://github.com/huggingface/transformers/tree/main/examples/pytorch/summarization#with-trainer

WANDB_MODE=offline \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "./scripts/run_summarization.py" "./conf/base.yml" "$CONFIG_FILEPATH" \
    output_dir="$OUTPUT_DIR" \
    perturbation="$PERTURBATION" \
    sampling_strategy="$SELECTION_STRATEGY" \
    perturbed_frac="$PERTURBED_FRAC" \
    perturbed_seed="$PERTURBED_SEED"