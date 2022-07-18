#!/bin/bash

####################################################################################################
# General setup for use on the Advanced Research Computing (ARC) clusters
# See https://alliancecan.ca/en/services/advanced-research-computing for cluster details.
####################################################################################################

# Load the required modules
# Notes: 
# - arrow needed for HF Datasets both during installation and use
# - rust is needed for HF Tokenizers library (and possibly other libraries) only during installation
# - java needed for Terrier (via PyTerrier) both during installation and use
module load StdEnv/2020 gcc/9.3.0 python/3.8 cuda/11.4 arrow/7.0.0 rust/1.59.0 java/11.0.2

# Setup the project and scratch directories
PROJECT_NAME="retrieval-exploration"
cd "$PROJECT/$USER"
git clone https://github.com/allenai/retrieval-exploration.git
cd "retrieval-exploration"
mkdir -p "$SCRATCH/$PROJECT_NAME"

# Create and activate the virtual environment
poetry shell
# poetry shell should activate the virtual environment, but it doesn't seem to? So do it manually:
# See https://python-poetry.org/docs/basic-usage/#activating-the-virtual-environment for details.
source $(poetry env info --path)/bin/activate

# Install the package
pip install --no-index --upgrade pip
# This will be faster in most cases
# See: https://github.com/python-poetry/poetry/issues/2094#issuecomment-605725577
# poetry export -f requirements.txt --dev --extras "summarization" --without-hashes > requirements.txt
# python -m pip install -r requirements.txt
# poetry install
poetry install --all-extras

cd "$PROJECT/$USER/retrieval-exploration"
