#!/bin/bash

####################################################################################################
# General setup for use on the Advanced Research Computing (ARC) clusters
# See https://alliancecan.ca/en/services/advanced-research-computing for cluster details.
####################################################################################################

# Load the required modules
# Note: rust is only need for compiling the tokenizers library
module load StdEnv/2020 gcc/9.3.0 python/3.8 cuda/11.4 rust/1.59.0 arrow/7.0.0

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
poetry install

# Install any task specific dependencies, e.g.
poetry install -E "summarization"


cd "$PROJECT/$USER/retrieval-exploration"