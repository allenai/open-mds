# Retrieval Exploration

[![ci](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/allenai/retrieval-exploration/branch/main/graph/badge.svg?token=YTQEI2VMSA)](https://codecov.io/gh/allenai/retrieval-exploration)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

## Installation

This repository requires Python 3.8 or later.

### Installing the library and dependencies

Many dependencies are task specific, so you will need to specify the correct task(s) via [`extras`](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-setuptools-extras). The lists of extras are:

- `"summarization"`: for summarization tasks
- `"all"`: for all tasks

#### Installing with pip

Install with `pip` right from GitHub, or clone the repo locally:

```bash
pip install "git+https://github.com/allenai/retrieval-exploration.git#egg=retrieval_exploration[all]"
```

or

```bash
git clone https://github.com/allenai/retrieval-exploration.git
cd retrieval-exploration
pip install -e ".[all]"
```

#### Installing with poetry

To install using [Poetry](https://python-poetry.org/):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for osx / linux / bashonwindows:
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

# Clone and move into the repo
git clone https://github.com/allenai/retrieval-exploration
cd retrieval-exploration

# Install the package with poetry
poetry install --all-extras

# To install only with task-specific dependencies, use the extras argument
poetry install --extras summarization
```

