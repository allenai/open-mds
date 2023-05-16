#!/bin/bash
# Requested resources
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
# ONLY required if using a dense retrieval pipeline
#SBATCH --gres=gpu:a100:1
# Wall time and job details
#SBATCH --time=12:00:00
#SBATCH --job-name=index
#SBATCH --account=def-wanglab-ab
# Use this command to run the same job interactively
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-wanglab-ab
# salloc --mem=32G --cpus-per-task=1 --gres=gpu:a100:1 --time=3:00:00 --account=def-gbader

### Example usage ###
# sbatch ./scripts/slurm/index.sh "multinews" "./output/datasets/multinews_sparse_oracle" "sparse" "oracle"
# sbatch ./scripts/slurm/index.sh "multinews" "./output/datasets/multinews_dense_mean" "dense" "mean"
# sbatch ./scripts/slurm/index.sh "ms2" "./output/datasets/ms2_sparse_max" "sparse" "max"
 
module purge  # suggested in alliancecan docs: https://docs.alliancecan.ca/wiki/Running_jobs
module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/8.0.0 java/14.0.2 cuda/11.4 faiss/1.7.1
PROJECT_NAME="open-mds"
source "$HOME/$PROJECT_NAME/bin/activate"
cd "$HOME/projects/def-gbader/$USER/$PROJECT_NAME" || exit

### Script arguments ###
# Required arguments
HF_DATASET_NAME="$1"    # The name of a supported HuggingFace Dataset
OUTPUT_DIR="$2"         # The path on disk to save the output to
RETRIEVER="$3"          # The type of retriever to use
STRATEGY="$4"           # The strategy to use when choosing the k top documents to retrieve

### Job ###
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
python "./scripts/index_and_retrieve.py" "$HF_DATASET_NAME" "$OUTPUT_DIR" \
    --retriever "$RETRIEVER" \
    --model-name-or-path "$HOME/projects/def-gbader/$USER/$PROJECT_NAME/output/models/contriever-msmarco" \
    --top-k-strategy "$STRATEGY" \
    --overwrite-cache

exit