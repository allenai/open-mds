# Reproducing the Experiments

This directory contains the SLURM submission scripts used to organize the experiments. This README serves the dual purpose of documenting the experiments for ourselves and providing instructions for others to reproduce them. Even if you are not using SLURM, they should help you to understand how the experiments were run. We also have a setup script ([`setup_on_arc.sh`](setup_on_arc.sh)) that was used to setup the project and environment on a [ARC](https://alliancecan.ca/en/services/advanced-research-computing) cluster which may be helpful.

> __Note__: Our experiments were run on a SLURM cluster without internet access. As a result, any model or data artifacts had to be downloaded and cached locally, and all libraries we used had to be prevented from accessing the internet (e.g. by using environment variables like `TRANSFORMERS_OFFLINE=1` or `HF_DATASETS_OFFLINE=1`). If you are also running on a SLURM cluster without internet access, you can use the [`cache.py`](cache.py) script to download and cache all the models and datasets we investigated in the paper. If you are running on a cluster with internet access, you can skip this step and use the names of the models and datasets from the HuggingFace Hub directly. If you are interested in reproducing the results from our paper, please follow the directions in this README. We also provide notebooks that, given the raw experimental results (available for download [here](https://ai2-s2-research-public.s3-us-west-2.amazonaws.com/open-mds/results.tar.gz)), reproduce the tables and figures from the paper. These notebooks can be found in the [notebooks](../../notebooks) directory.

## Open-domain MDS Experiments

The open-domain MDS experiments are reproduced with the `retrieval.sh` and `submit_retrieval.sh` scripts.

> To build the document indices from scratch and retrieve the input documents for each dataset, see [Index and Retrieve the Input Documents](#optional-index-and-retrieve-the-input-documents).

### [`retrieve.sh`](retrieve.sh)

This script is used to run a single retrieval experiment, e.g.

```bash
sbatch "./scripts/slurm/retrieve.sh" "./conf/multinews/primera/eval.yml" \
  "./output/results/multinews/primera/retrieval/sparse/mean" \
  "allenai/multinews_sparse_mean" \
  "sparse" \
  "mean"
```

would submit a job evaluating the [PRIMERA](https://arxiv.org/abs/2110.08499) model on the [Multi-News](https://aclanthology.org/P19-1102/) dataset, whose documents have been retrieved using the `"sparse"` retriever with the `"mean"` top-k strategy, and save the results to `"./output/results/multinews/primera/retrieval/sparse/mean"`.

### [`submit_retrieve.sh`](submit_retrieve.sh)

This script is used to submit _all_ retrieval experiments for a given model-dataset pair, e.g.

```bash
bash "./scripts/slurm/submit_retrieve.sh" "./conf/multinews/primera/eval.yml" \
  "./output/results/multinews/primera" \
  "./output/datasets/multinews"
```

would submit all retrieval experiments for the [PRIMERA](https://arxiv.org/abs/2110.08499) model on the [Multi-News](https://aclanthology.org/P19-1102/) dataset and save the results to `"./output/results/multinews/primera/retrieval"`.

### (OPTIONAL) Index and Retrieve the Input Documents

This is only required if you want to re-index the datasets and re-retrieve the input documents. We have already done this and made them publicly available. Just look for `"allenai/[dataset_name]_[sparse|dense]_[max|mean|oracle]"` on [HuggingFace](https://huggingface.co/datasets), e.g. [`"allenai/multinews_sparse_mean"`](https://huggingface.co/datasets/allenai/multinews_sparse_mean).

#### [`index.sh`](index.sh)

This script is used to index a dataset and replace its documents with those retrieved from the index, using a single retriever and top-k strategy, e.g.

```bash
sbatch ./scripts/slurm/index.sh "multinews" "./output/datasets/multinews_sparse_oracle" "sparse" "oracle"
```

would index the [Multi-News](https://aclanthology.org/P19-1102/) dataset, replace its documents with those retrieved from the index by a sparse retriever, and save the resulting dataset to `"./output/datasets/multinews_sparse_oracle"`. Note that this is actually calling the [`index_and_retrieve.py`](../../scripts/index_and_retrieve.py) script, which you can get more information about by calling

```bash
python ./scripts/index_and_retrieve.py --help
```

#### [`submit_index.sh`](submit_index.sh)

Submits an array of jobs for a given dataset, producing multiple copies where each contains its documents replaced by those retrieved from the index using a different retriever and top-k strategy, e.g.

```bash
bash "./scripts/slurm/submit_index.sh" "multinews" "./output/datasets" 
```

would all jobs for the [Multi-News](https://aclanthology.org/P19-1102/) dataset, creating multiple copies of the dataset using all retrievers and top-k strategies, and save the resulting datasets to `"./output/datasets"`.

## Extended Training Experiments

The extended training experiments are reproduced with the [`retrieve.sh`](retrieve.sh) and `submit_eval_checkpoints` scripts.

### [`retrieve.sh`](retrieve.sh)

This script is used to extend the training of the model on the retrieved documents. It is used in a similar way as in the retrieval experiments, except we provide a different configuration file (`"train_retrieved.yml"`), e.g.

```bash
sbatch "./scripts/slurm/retrieve.sh" "./conf/multinews/primera/train_retrieved.yml" \
  "./output/results/multinews/primera/trained_with_retrieval" \
  "allenai/multinews_dense_mean" \
  "dense" \
  "mean"
```

would submit a job to train the [PRIMERA](https://arxiv.org/abs/2110.08499) model on the [Multi-News](https://aclanthology.org/P19-1102/) dataset, whose documents have been retrieved using the `"dense"` retriever with the `"mean"` top-k strategy, and save the results to `"./output/results/multinews/primera/trained_with_retrieval"`.

### [`submit_eval_checkpoints.sh`](submit_eval_checkpoints.sh)

This script will evaluate all the checkpoints from a single training run generated by [`retrieve.sh`](retrieve.sh) in the previous step.

```bash
bash "./scripts/slurm/submit_eval_checkpoints.sh" \
  "./conf/multinews/primera/eval.yml" \
  "./output/results/multinews/primera/trained_with_retrieval" \
  "./output/results/multinews/primera" \
  "allenai/multinews_dense_mean" \
  "dense" \
  "mean"
```

would submit jobs to evaluate each of the trained checkpoints in `"./output/multinews/primera/trained_with_retrieval"` and save the results to `"./output/results/multinews/primera/training"`. Note that this is calling two scripts, `eval_checkpoint_retrieved.sh` and `eval_checkpoint_gold.sh`, which aren't expected to be called directly.

## Perturbation Experiments

The perturbation experiments are reproduced with the [`perturb.sh`](perturb.sh) and [`submit_perturb.sh`](submit_perturb.sh) scripts.

### [`perturb.sh`](perturb.sh)

This script is used to run a single perturbation experiment, e.g.

```bash
sbatch "./scripts/slurm/perturb.sh" "./conf/multinews/primera/eval.yml" \
  "./output/results/multinews/primera/peturbations/random/addition/0.1" \
  "addition" \
  "random" \
  "0.1"
```

would submit a job evaluating the [PRIMERA](https://arxiv.org/abs/2110.08499) model on the [Multi-News](https://aclanthology.org/P19-1102/) dataset, subject to the `"addition"` perturbation experiment with the `"random"` strategy (perturbing 10% of the input documents) and save the results to `"./output/results/multinews/primera/peturbations/random/addition/0.1"`.

### [`submit_perturb.sh`](submit_perturb.sh)

This script is used to submit _all_ perturbation experiments for a given model-dataset pair, including the baseline evaluation (no perturbations), e.g.

```bash
bash "./scripts/slurm/submit_perturb.sh" \
  "./conf/multinews/primera/eval.yml" \
  "./output/results/multinews/primera"
```

would submit all perturbation experiments for the [PRIMERA](https://arxiv.org/abs/2110.08499) model on the [Multi-News](https://aclanthology.org/P19-1102/) dataset (including the baseline evaluation) and save the results to `"./output/results/multinews/primera/peturbations"`.

## Uploading the data to S3

Once all the results were generated, we compressed the directory with the following command:

```bash
tar cvfz output.tar.gz \
  --exclude '.DS_Store' \
  --exclude 'README.md' \
  --exclude 'plots/*' \
  --exclude 'tables/*' \
  --exclude '*/eval_results.json' \
  --exclude '*/predict_results.json' \
  --exclude '*/generated_predictions.txt' \
  --exclude '*/runs/*' \
  --exclude '*/trained_with_retrieval/*' \
  output
```

and uploaded the resulting `tar` file to an AWS S3 bucket

```bash
aws s3 cp "output.tar.gz" "s3://ai2-s2-research-public/open-mds/output.tar.gz"
```