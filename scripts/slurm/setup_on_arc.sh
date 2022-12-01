#!/bin/bash

####################################################################################################
# General setup for use on the Advanced Research Computing (ARC) clusters
# See https://alliancecan.ca/en/services/advanced-research-computing for cluster details.
####################################################################################################

# Load the required modules
# Notes: 
# - arrow needed for HF Datasets both during installation and use
# - java needed for Terrier (via PyTerrier) both during installation and use
module purge
module load StdEnv/2020 gcc/9.3.0 python/3.9 arrow/7.0.0 java/14.0.2

# Setup the virtual environment under home
PROJECT_NAME="open-mds"
virtualenv --no-download "$HOME/$PROJECT_NAME"
source "$HOME/$PROJECT_NAME/bin/activate"
pip install --no-index --upgrade pip

# Setup the project and scratch directories
# NOTE: On some clusters (e.g. Narval), the PROJECT env var does not exist, so you will have to cd manually
cd "$PROJECT/$USER" || exit
git clone https://github.com/allenai/open-mds.git
cd "open-mds" || exit
mkdir -p "$SCRATCH/$PROJECT_NAME"

# Install the package
# The package is built with poetry, so make sure that it is available first
pip install "git+https://github.com/python-poetry/poetry.git"
poetry -V
# Unfortunately, poetry install causes a host of issues on ARC machines
# As a workaround, install only the package with poetry, and all of its dependencies with pip
poetry install --only-root
# For several packages, there are less errors when installing the pre-built wheels
pip install scipy numpy pandas torch torchvision --no-index
# We also need the latest version of transformers, so install from git
pip install "git+https://github.com/huggingface/transformers.git"
# Lastly, we can export the project's dependencies to a requirements file and install with pip
poetry export -f requirements.txt --output requirements.txt -E "retrieval" --without-hashes
# but first, remove any of the dependencies we are installing manually
sed -i '/^scipy\|^numpy\|^pandas\|^torch\|^torchvision\|^transformers\|^pyarrow/d' requirements.txt 
pip install -r requirements.txt
