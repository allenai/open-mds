# Retrieval Exploration

[![ci](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/allenai/retrieval-exploration/branch/main/graph/badge.svg?token=YTQEI2VMSA)](https://codecov.io/gh/allenai/retrieval-exploration)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

## Installation

This repository requires Python 3.8 or later.

### Installing the library and dependencies

Many dependencies are task-specific, so you will need to specify the correct task(s) via [`extras`](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-setuptools-extras). The lists of extras are:

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

## Usage

This repository provides a collection of modified `run_*.py` scripts from HuggingFace. The intention is to study the resilience of multi-document models to _document-level_ perturbations. In particular, these perturbations mimic errors that a _retriever_ might make:

- `"sorting"`: Shuffle the order of input documents to simulate different rank-ordered lists from a retriever.
- `"duplication"`: Duplicate input documents to simulate the retrieval of duplicate documents from a large corpus.
- `"addition"`: Add one or more documents to the input to mimic the retrieval of an irrelevant document. 
- `"deletion"`: Remove one or more documents from the input to mimic the failure to retrieve a relevant document.
- `"replacement"`: Replace one or more documents in the input with another document. This is a combination of `"addition"` and `"deletion"`.

We include three different "strategies" that apply to each perturbation:

- `"random"`: Randomly selects documents for each perturbation.
- `"best-case"`: Attempts to select documents in a way that is expected to minimize the harm of the perturbation. E.g. for `"addition"` in summarization, documents are selected based on similarity to the target summary.
- `"worst-case"`: Attempts to select documents in a way that is expected to maximize the harm of the perturbation. E.g. for `"addition"` in summarization, documents are selected based on dissimilarity to the target summary.

Usage is similar to the `run_*.py` scripts from HuggingFace, but with extra arguments for the perturbation. To make things easier, we provide configs for several popular models and datasets. Here are a few examples:

Evaluate [PEGASUS](https://arxiv.org/abs/1912.08777) with the `"deletion"` perturbation and `"random"` strategy, perturbing 10% of input documents, on the [Multi-News](https://aclanthology.org/P19-1102/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multinews/pegasus/eval.yml" \
    --output_dir "./output/multinews/pegasus/eval/perturbations/deletion/random/0.10" \
    --perturbation "deletion" \
    --strategy "random" \
    --perturbed_frac 0.10
```

Train [PRIMERA](https://arxiv.org/abs/2110.08499) with the `"addition"` perturbation and `"best-case"` strategy, perturbing 50% of input documents, on the [Multi-XScience](https://aclanthology.org/2020.emnlp-main.648/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multixscience/primera/train.yml" \
    --output_dir "./output/multixscience/primera/train/perturbations/addition/best-case/0.50" \
    --perturbation "addition" \
    --strategy "best-case" \
    --perturbed_frac 0.50
```

Other experiments can be crafted by modifying the perturbation arguments accordingly.

### Notes

In order to avoid duplicated computation, some perturbations will cache their results. You can get this path by calling

```bash
python -c "from retrieval_exploration.common import util ; print(util.CACHE_DIR)"
```