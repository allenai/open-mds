#!/bin/bash

### Example usage ###
# bash "./scripts/slurm/submit_retrieval.sh" "./conf/multinews/primera/eval.yml" \
#     "./output/multinews/primera" \
#     "./output/datasets/multinews" \

### Script arguments ###
# Must be provided as argument to the script
CONFIG_FILEPATH="$1"  # The path on disk to the yml config file
OUTPUT_DIR="$2"       # The path on disk to save the output to
DATASET_DIR="$3"      # The path on disk to the dataset to use
# Constants
RETRIEVERS=("sparse")
STRATEGIES=("mean" "max" "oracle")

### Job ###

# Run the grid
for retriever in "${RETRIEVERS[@]}";
do
    for strategy in "${STRATEGIES[@]}";
    do
        sbatch "./scripts/slurm/retrieval.sh" "$CONFIG_FILEPATH" \
            "$OUTPUT_DIR/perturbations/retrieval/$retriever/$strategy" \
            "${DATASET_DIR}_${retriever}_${strategy}" \
            "${retriever}" \
            "${strategy}"
    done
done