# Open-Domain Multi-Document Summarization

[![ci](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/retrieval-exploration/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/allenai/retrieval-exploration/branch/main/graph/badge.svg?token=YTQEI2VMSA)](https://codecov.io/gh/allenai/retrieval-exploration)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)

The intention of this repository is to study end-to-end performance of __multi-document summarization__ (MDS) models, that is, when the input document set must be retrieved via (semi-)automatic methods.

We do this via two mechanisms. First by subjecting MDS models to a suite of peturbations designed to mimic retrieval errors (see [Perturbations](#perturbations)), and secondly by using retrieval to re-build the input document sets of MDS datasets (see [Retrieval](#retrieval)) and evaluating MDS models on these datasets.

## Installation

This repository requires Python 3.8 or later.

### Installing with pip

Install with `pip` right from GitHub

```bash
pip install "git+https://github.com/allenai/retrieval-exploration.git"
```

or clone the repo locally

```bash
git clone https://github.com/allenai/retrieval-exploration.git
cd retrieval-exploration
pip install -e .
```

### Installing with poetry

To install using [Poetry](https://python-poetry.org/)

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and move into the repo
git clone https://github.com/allenai/retrieval-exploration
cd retrieval-exploration

# Install the package with poetry
poetry install
```

## Usage

### Perturbations

The available perturbations mimic errors that a __retriever__ might make:

- `"sorting"`: Shuffle the order of input documents to simulate different rank-ordered lists from a retriever.
- `"duplication"`: Duplicate input documents to simulate the retrieval of duplicate (or near-duplicate) documents from a large corpus.
- `"addition"`: Add one or more documents to the input to mimic the retrieval of an irrelevant document. 
- `"deletion"`: Remove one or more documents from the input to mimic the failure to retrieve a relevant document.
- `"replacement"`: Replace one or more documents in the input with another document. This is a combination of `"addition"` and `"deletion"`.
- `"backtranslation"`: Replace one or more documents in the input with a backtranslated copy. This mainly serves as a control.

We include three different "selection strategies" that apply to each perturbation:

- `"random"`: Randomly selects documents for each perturbation.
- `"best-case"`: Attempts to select documents in a way that is expected to _minimize_ the harm of the perturbation. E.g. for `"deletion"`, documents _least similar_ to the target summary are removed first.
- `"worst-case"`: Attempts to select documents in a way that is expected to _maximize_ the harm of the perturbation. E.g. for `"deletion"`, documents _most similar_ to the target summary are removed first.

Usage is similar to the original `run_summarization*.py` script from HuggingFace, but with extra arguments for the perturbation. To make things easier, [we provide configs](./conf) for several popular models and datasets. Here are a few examples:

Evaluate [PEGASUS](https://arxiv.org/abs/1912.08777) with the `"deletion"` perturbation and `"random"` strategy, perturbing 10% of input documents, on the [Multi-News](https://aclanthology.org/P19-1102/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multinews/pegasus/eval.yml" \
    --output_dir "./output/multinews/pegasus/eval/perturbations/deletion/random/0.10" \
    --perturbation "deletion" \
    --strategy "random" \
    --perturbed-frac 0.10
```

Evaluate [PRIMERA](https://arxiv.org/abs/2110.08499) with the `"addition"` perturbation and `"best-case"` strategy, perturbing 50% of input documents, on the [Multi-XScience](https://aclanthology.org/2020.emnlp-main.648/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multixscience/primera/train.yml" \
    --output_dir "./output/multixscience/primera/train/perturbations/addition/best-case/0.50" \
    --perturbation "addition" \
    --strategy "best-case" \
    --perturbed-frac 0.50
```

Other experiments can be crafted by modifying the perturbation arguments accordingly. New models and datasets can be added by creating your own YAML based config file. See [conf](conf) for examples.

#### Notes

- In order to avoid duplicate computation, some perturbations will cache their results. You can get this path by calling

  ```bash
  python -c "from retrieval_exploration.common import util ; print(util.CACHE_DIR)"
  ```

### Retrieval

We also provide a script, [index_and_retrieve.py](./scripts/index_and_retrieve.py), for re-building the input document sets of several popular MDS datasets with retrieval. To run the script, make sure you have installed the required dependencies

```bash
# With pip
pip install "git+https://github.com/allenai/retrieval-exploration.git#egg=retrieval_exploration[retrieval]"

# OR, if installin with poetry
poetry install -E "retrieval"
```

Then you can see detailed instructions by calling

```bash
python ./scripts/index_and_retrieve.py --help
```

Here are a few examples:

Re-build the `"test"` set of the [Multi-News](https://aclanthology.org/P19-1102/) dataset with a `"sparse"` retriever, using the `"oracle"` strategy to choose `k`

```bash
python ./scripts/index_and_retrieve.py "multinews" "./output/datasets/multinews_sparse_oracle" \
    --retriever "sparse" \
    --top-k-strategy "oracle" \
    --splits "test"
```

Re-build the `"validation"` and test set of the [MS^2](https://aclanthology.org/2021.emnlp-main.594/) dataset with a `"dense"` retriever, using the `"mean"` strategy to choose `k`

```bash
python ./scripts/index_and_retrieve.py "ms2" "./output/datasets/ms2_dense_mean" \
    --retriever "dense" \
    --top-k-strategy "mean" \
    --splits "validation"
```

#### Notes

- We have re-built several popular MDS datasets with retrieval and made them publically available. On the [HuggingFace Hub](https://huggingface.co/). Just search for `"allenai/[dataset_name]_[sparse|dense]_[max|mean|oracle]"`, e.g. [allenai/multinews_dense_max](https://huggingface.co/datasets/allenai/multinews_dense_max).
- If `index-path` is not provided, document indices will be saved to disk under a default location. You can get this dock path by calling

  ```bash
  python -c "from retrieval_exploration.common import util ; print(util.CACHE_DIR)"
  ```

- If you wish to use the `dense` retriever, you will need to install [FAISS](https://github.com/facebookresearch/faiss) with GPU support. See [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for detailed instructions.

## Reproducing Results

If you are interested in reproducing the results from our paper, please see the README in the [scripts/slurm](./scripts/slurm) directory. We also provide notebooks that, given the raw experimental results, reproduce the tables and figures from the paper. These notebooks can be found in the [notebooks](./notebooks) directory.