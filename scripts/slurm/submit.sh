#!/bin/bash
# Must be provided as argument to the script
CONFIG_FILEPATH="$1"
# Constants
PERTURBATIONS=("backtranslation" "duplication" "addition" "deletion" "replacement")
STRATEGIES=("random" "best-case" "worst-case")
PERTURBED_FRAC=(0.1 0.5 1.0)
PERTURBED_SEED=42

# Run the grid
for perturbation in "${PERTURBATIONS[@]}";
do
    for strategy in "${STRATEGIES[@]}";
    do
        # Sorting does not need to run for multiple perturbed fractions
        sbatch scripts/run_summarization.py "conf/base.yml" "$CONFIG_FILEPATH" \
            perturbation="sorting" \
            sampling_strategy="$strategy" \
            perturbed_seed=$PERTURBED_SEED

        for perturbed_frac in "${PERTURBED_FRAC[@]}";
        do
            sbatch scripts/run_summarization.py "conf/base.yml" "$CONFIG_FILEPATH" \
                perturbation="${perturbation}" \
                perturbed_frac="${perturbed_frac}" \
                perturbed_seed=$PERTURBED_SEED
        done
    done
done