#!/bin/bash

### Example usage ###
# bash "./scripts/slurm/submit_index.sh" "multinews" "./output/datasets"

### Script arguments ###
# Must be provided as argument to the script
HF_DATASET_NAME="$1"
OUTPUT_DIR="$2"
# Constants
RETRIEVERS=("sparse")
STRATEGIES=("mean" "max" "oracle")
SPLITS="test"

### Job ###

# Run the grid
for retriever in "${RETRIEVERS[@]}";
do
    for strategy in "${STRATEGIES[@]}";
    do
        sbatch "./scripts/slurm/index.sh" "$HF_DATASET_NAME" \
            "$OUTPUT_DIR/${HF_DATASET_NAME}_${retriever}_$strategy" \
            "${retriever}" \
            "${strategy}" \
            "$SPLITS"
    done
done