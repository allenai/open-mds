# Open-Domain Multi-Document Summarization

[![ci](https://github.com/allenai/open-mds/actions/workflows/ci.yml/badge.svg)](https://github.com/allenai/open-mds/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/allenai/open-mds/branch/main/graph/badge.svg?token=YTQEI2VMSA)](https://codecov.io/gh/allenai/open-mds)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
![GitHub](https://img.shields.io/github/license/allenai/open-mds?color=blue)

The corresponding code for our paper: Exploring the Challenges of Open Domain Multi-Document Summarization. The raw experimental results are available for download [here](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/open-mds/results.tar.gz).

## Installation

This repository requires Python 3.8 or later.

### Installing with pip

First, activate a virtual environment. Then, install with `pip` right from GitHub:

```bash
pip install "git+https://github.com/allenai/open-mds.git"
```

or clone the repo locally and install from source:

```bash
git clone https://github.com/allenai/open-mds.git
cd open-mds
pip install -e .
```

### Installing with poetry

To install using [Poetry](https://python-poetry.org/) (this will activate a virtual environment for you):

```bash
# Install poetry for your system: https://python-poetry.org/docs/#installation
# E.g. for Linux, macOS, Windows (WSL)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and move into the repo
git clone https://github.com/allenai/open-mds
cd open-mds

# Install the package with poetry
poetry install
```

## Usage

There are several ways to interact with this codebase. If you are interested in analyzing the results of our experiments or reproducing tables or figures from the paper, please see [📚 Notebooks](#📚-notebooks). If you are interested in running the open-domain MDS experiments, see [🔎 Open-domain MDS](#🔎-open-domain-mds). If you are interested in running our experiments simulating document retrieval errors, see [🧪 Simulating Document Retrieval Errors](#🧪-simulating-document-retrieval-errors). If you would like to reproduce the results in the paper from scratch, please see our detailed instructions [here](./scripts/slurm).

### 📚 Notebooks

We have notebooks corresponding to each of the major experiments in the paper:

- [Dataset Statistics](./notebooks/dataset_statistics.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/dataset_statistics.ipynb)): Compute simple dataset statistics for each dataset in the paper.
- [Open-Domain MDS](./notebooks/open-mds.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/open-mds.ipynb)): Analyze the results from the open-domain MDS experiments.
- [Baselines](./notebooks/baselines.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/baselines.ipynb)): Computes the summarization performance of several simple baselines for each dataset in the paper.
- [Training](./notebooks/training.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/training.ipynb)): Analyze the results from the experiments where we fine-tune summarizers in the open-domain setting.
- [Perturbations](./notebooks/perturbations.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/perturbations.ipynb)): Analyze the results of our simulated document retrieval error experiments.
- [Sorting](./notebooks/sorting.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/allenai/open-mds/blob/main/notebooks/sorting.ipynb)): Analyze the results of the sorting perturbation experiment.

If you are running the notebooks locally, make sure to add a virtual environment with this project installed in it to an IPython kernel:

```bash
pip install --user ipykernel
python -m ipykernel install --user --name=<myenv>
```

You can now select this environment as a kernel in the notebook. See [here](https://janakiev.com/blog/jupyter-virtual-envs/#add-virtual-environment-to-jupyter-notebook) for more details.

> Note: some IDEs, like VSCode, will automate this process when you launch a notebook with the virtual environment active.

### 🔎 Open-domain MDS

In the paper, we bootstrap the newly proposed task of open-domain MDS using existing datasets, retrievers, and summarizers. To run these experiments, there are two steps:

#### 1️⃣ (OPTIONAL) Index and Retrieve the Input Documents

This is only required if you want to re-index the datasets and re-retrieve the input documents. We have already done this and made them publicly available. Just look for `"allenai/[dataset_name]_[sparse|dense]_[max|mean|oracle]"` on [HuggingFace](https://huggingface.co/datasets), e.g. [`"allenai/multinews_sparse_mean"`](https://huggingface.co/datasets/allenai/multinews_sparse_mean).

Otherwise, please use the [index_and_retrieve.py](./scripts/index_and_retrieve.py) script. First, make sure you have installed the required dependencies

```bash
# With pip
pip install "git+https://github.com/allenai/open-mds.git#egg=open_mds[retrieval]"

# OR, if installing with poetry
poetry install -E "retrieval"
```

Then you can see detailed instructions by calling

```bash
python ./scripts/index_and_retrieve.py --help
```

Here are a few examples:

Re-build the examples of the [Multi-News](https://aclanthology.org/P19-1102/) dataset with a `"sparse"` retriever and `"oracle"` top-`k` strategy

```bash
python ./scripts/index_and_retrieve.py "multinews" "./output/datasets/multinews_sparse_oracle" \
    --retriever "sparse" \
    --top-k-strategy "oracle"
```

Re-build the examples of the [MS^2](https://aclanthology.org/2021.emnlp-main.594/) dataset with a `"dense"` retriever and `"mean"` top-`k` strategy

```bash
python ./scripts/index_and_retrieve.py "ms2" "./output/datasets/ms2_dense_mean" \
    --retriever "dense" \
    --top-k-strategy "mean"
```

Other experiments can be crafted by modifying the arguments accordingly.

##### Notes

- If `index-path` is not provided, document indices will be saved to disk under a default location. You can get this path by calling

  ```bash
  python -c "from open_mds.common import util ; print(util.CACHE_DIR)"
  ```

- If you wish to use the `dense` retriever, you will need to install [FAISS](https://github.com/facebookresearch/faiss) with GPU support. See [here](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) for detailed instructions.

#### Evaluate Summarizers in the Open-Domain Setting

Usage is similar to the original [`run_summarization.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py) script from HuggingFace, but with extra arguments for the retrieval. To make things easier, [we provide configs](./conf) for the models and datasets we investigated. Here are a few examples:

1️⃣ Evaluate [PEGASUS](https://arxiv.org/abs/1912.08777) with a `"sparse"` retriever and `"mean"` top-`k` strategy on the [Multi-News](https://aclanthology.org/P19-1102/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multinews/pegasus/eval.yml" \
    output_dir="./output/multinews/pegasus/retrieval/sparse/mean" \
    dataset_name="allenai/multinews_sparse_mean" \
    retriever="sparse" \
    top_k_strategy="mean"
```

2️⃣ Evaluate [PRIMERA](https://arxiv.org/abs/2110.08499) with a `"dense"` retriever and `"oracle"` top-`k` strategy on the [Multi-XScience](https://aclanthology.org/2020.emnlp-main.648/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multixscience/primera/eval.yml" \
    output_dir="./output/multixscience/primera/retrieval/dense/oracle" \
    dataset_name="allenai/multixscience_dense_oracle" \
    retriever="dense" \
    top_k_strategy="oracle"
```

Other experiments can be crafted by modifying the arguments accordingly.

### 🧪 Simulating Document Retrieval Errors

In the paper, we simulate document retrieval errors by perturbing the input document sets of several popular MDS datasets _before_ they are provided to the summarizer. Each of the perturbations is designed to mimic an error likely to be made by a __retriever__ in the open-domain setting:

- `"addition"`: Add one or more documents to the input to mimic the retrieval of irrelevant documents. 
- `"deletion"`: Remove one or more documents from the input to mimic the failure to retrieve relevant documents.
- `"duplication"`: Duplicate input documents to simulate the retrieval of duplicate (or near-duplicate) documents from the index.
- `"replacement"`: Replace one or more documents in the input with another document. This is a combination of `"addition"` and `"deletion"`.
- `"sorting"`: Sort (or shuffle) the order of input documents to simulate different rank-ordered lists from a retriever.
- `"backtranslation"`: Replace one or more documents in the input with a backtranslated copy. This is not an error a retriever would make, but allows us to compare and contrast the known sensitivity of NLP models to small token-level changes in their inputs with the document-level changes we are interested in.

We include two different document "selection strategies" that apply to each perturbation:

- `"random"`: Randomly selects documents for each perturbation. This mimics a (very) *weak* retriever.
- `"oracle"`: Attempts to select documents in a way that mimics a *strong* retriever. E.g. for `"deletion"`, documents _least similar_ to the target summary are removed first.

> Please see the paper for more details on the experimental setup.

Usage is similar to the original [`run_summarization.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/summarization/run_summarization.py) script from HuggingFace, but with extra arguments for the perturbation. To make things easier, [we provide configs](./conf) for the models and datasets we investigated. Here are a few examples:

1️⃣ Evaluate [PEGASUS](https://arxiv.org/abs/1912.08777) with the `"deletion"` perturbation and `"random"` document selection strategy, perturbing 10% of input documents on the [Multi-News](https://aclanthology.org/P19-1102/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multinews/pegasus/eval.yml" \
    output_dir="./output/multinews/pegasus/perturbations/deletion/random/0.10" \
    perturbation="deletion" \
    selection_strategy="random" \
    perturbed_frac 0.10
```

2️⃣ Evaluate [PRIMERA](https://arxiv.org/abs/2110.08499) with the `"addition"` perturbation and `"oracle"` strategy, perturbing 50% of input documents on the [Multi-XScience](https://aclanthology.org/2020.emnlp-main.648/) dataset

```bash
python ./scripts/run_summarization.py "./conf/base.yml" "./conf/multixscience/primera/eval.yml" \
    output_dir="./output/multixscience/primera/perturbations/addition/oracle/0.50" \
    perturbation="addition" \
    selection_strategy="oracle" \
    perturbed_frac 0.50
```

Other experiments can be crafted by modifying the arguments accordingly.

#### Notes

- To avoid duplicate computation, some perturbations (like backtranslation) will cache their results. You can get this path by calling

  ```bash
  python -c "from open_mds.common import util ; print(util.CACHE_DIR)"
  ```