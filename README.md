# Retrieval Exploration

## Installation

This repository requires Python 3.8 or later.

### Installing the library and dependencies

If you _do not_ plan on modifying the source code, install from `git` using `pip`

```bash
pip install git+https://github.com/allenai/retrieval-exploration.git
```

Otherwise, clone the repository and install from source using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for osx / linux / bashonwindows:
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/allenai/retrieval-exploration
cd seq2rel

# Install the package with poetry
poetry install
```

