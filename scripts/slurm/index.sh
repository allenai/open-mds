#!/bin/bash
# Requested resources
#SBATCH --mem=96G
#SBATCH --cpus-per-task=1
# Wall time and job details
#SBATCH --time=5:00:00
#SBATCH --job-name=index
#SBATCH --account=def-wanglab-ab
# Emails me when job starts, ends or fails
#SBATCH --mail-user=johnmgiorgi@gmail.com
#SBATCH --mail-type=FAIL
# Use this command to run the same job interactively
# salloc --mem=96G --cpus-per-task=1 --time=3:00:00 --account=def-wanglab-ab
# salloc --mem=96G --cpus-per-task=1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sbatch ./scripts/slurm/retrieve.sh "multinews" "./output/datasets/multinews_sparse_oracle" "sparse" "oracle" "test"
 
module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/7.0.0 java/14.0.2
PROJECT_NAME="retrieval-exploration"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/def-gbader/$USER/$PROJECT_NAME" || exit

### Script arguments ###
# Must be provided as argument to the script
HF_DATASET_NAME="$1"    # The name of a supported HuggingFace Dataset
OUTPUT_DIR="$2"         # The path on disk to save the output to
RETRIEVER="$3"          # The type of retriever to use
STRATEGY="$4"           # The strategy to use when choosing the k top documents to retrieve
SPLITS="$5"             # Which splits of the dataset to replace with retrieved documents 

### Job ###

HF_DATASETS_OFFLINE=1 \
python "./scripts/retrieval.py" "$HF_DATASET_NAME" "$OUTPUT_DIR" \
    --retriever "$RETRIEVER" \
    --top-k-strategy "$STRATEGY" \
    --splits "$SPLITS" \
    --overwrite-cache
