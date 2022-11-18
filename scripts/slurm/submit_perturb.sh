#!/bin/bash

### Example usage ###
# bash "./scripts/slurm/submit_perturb.sh" \
# "./conf/multinews/primera/eval.yml" \
# "./output/results/multinews/primera"

### Usage notes ###
# Some perturbations take much longer than others (e.g. addition vs deletion). It is therefore advisable to submit
# this script multiple times, with different job runtimes depending on the perturbation. Additionally, some
# perturbations cache intermediate results (backtranslation). Tt is advisable to run these perturbations
# sequentially with an increasing PERTURBED_FRAC.

### Script arguments ###
# Required arguments
CONFIG_FILEPATH="$1"  # The path on disk to the yml config file
OUTPUT_DIR="$2"       # The path on disk to save the output to
# Constants
PERTURBATIONS=("backtranslation" "duplication" "addition" "deletion" "replacement")
STRATEGIES=("random" "oracle")
PERTURBED_FRAC=(0.1 0.25 0.5 0.75 1.0)

### Job ###

# Run the baseline
sbatch "./scripts/slurm/perturb.sh" "$CONFIG_FILEPATH" "$OUTPUT_DIR/baseline"

# Run the grid
for strategy in "${STRATEGIES[@]}";
do
    # Sorting does not need to run for multiple perturbed fractions
    sbatch "./scripts/slurm/perturb.sh" "$CONFIG_FILEPATH" \
        "$OUTPUT_DIR/perturbations/$strategy/sorting" \
        "sorting" \
        "${strategy}"

    for perturbation in "${PERTURBATIONS[@]}";
    do
        for perturbed_frac in "${PERTURBED_FRAC[@]}";
        do
            sbatch "./scripts/slurm/perturb.sh" "$CONFIG_FILEPATH" \
                "$OUTPUT_DIR/perturbations/$strategy/$perturbation/$perturbed_frac" \
                "${perturbation}" \
                "${strategy}" \
                "${perturbed_frac}"
        done
    done
done