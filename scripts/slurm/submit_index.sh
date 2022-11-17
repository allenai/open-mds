#!/bin/bash

### Example usage ###
# bash "./scripts/slurm/submit_index.sh" "multinews" "./output/datasets" 

### Usage notes ###
# If a dataset needs to be indexed, running this script for multiple strategies can lead to multiple jobs running
# simultaneously, all trying to create the index and save it to disk at the same location, which is error-prone.
# Therefore, it is advised to run the script once for a single retriever (say, "sparse") and a single strategy
# (say, "mean"), allowing the index to be created, and then re-run the script for the remaining strategies.
#
# The larger datasets (e.g. those with 10s of thousands of example like multinews and ms2) can take around 12
# hours to perform the index, retireval and rewriting steps. The smaller datasets (those with thousands or
# hundreds of examples like wcep and cochrane) take considerably less time (e.g. < 3 hours).

### Script arguments ###
# Required arguments
HF_DATASET_NAME="$1"  # The name of a supported HuggingFace Dataset
OUTPUT_DIR="$2"       # The path on disk to save the output to
# Optional arguments
RETRIEVERS=("sparse" "dense")
STRATEGIES=("max" "mean" "oracle")

### Job ###
# Run the grid
for retriever in "${RETRIEVERS[@]}";
do
    for strategy in "${STRATEGIES[@]}";
    do
        sbatch "./scripts/slurm/index.sh" "$HF_DATASET_NAME" \
            "$OUTPUT_DIR/${HF_DATASET_NAME}_${retriever}_${strategy}" \
            "${retriever}" \
            "${strategy}"
    done
done