#!/bin/bash

### Example usage ###
# bash "./scripts/slurm/submit_eval_checkpoints.sh" \
#   "./conf/multinews/primera/eval.yml" \
#   "./output/results/multinews/primera/trained_with_retrieval" \
#   "./output/results/multinews/primera" \
#   "./output/datasets/multinews_dense_mean" \
#   "dense" \
#   "mean"

### Script arguments ###
# Required arguments
CONFIG_FILEPATH="$1"  # The path on disk to the yml config file
CHECKPOINT_DIR="$2"   # The path on disk to the trained checkpoints
OUTPUT_DIR="$3"       # The path on disk to save the output to
DATASET_DIR="$4"      # The path on disk to the dataset to use
RETRIEVER="$5"        # The type of retriever to use
STRATEGY="$6"         # The strategy to use when choosing the k top documents to retrieve

for dir in $CHECKPOINT_DIR/checkpoint-*/; do
    checkpoint=$(basename "$dir")

    # Evaluate on the retrieved test set
    sbatch "./scripts/slurm/eval_checkpoint_retrieved.sh" "$CONFIG_FILEPATH" \
        "$OUTPUT_DIR/training/retrieved/$checkpoint" \
        "$dir" \
        "$DATASET_DIR" \
        "$RETRIEVER" \
        "$STRATEGY"

    # Evaluate on the gold test set
    sbatch "./scripts/slurm/eval_checkpoint_gold.sh" "$CONFIG_FILEPATH" \
        "$OUTPUT_DIR/training/gold/$checkpoint" \
        "$dir"
done