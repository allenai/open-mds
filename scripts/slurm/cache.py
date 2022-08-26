#!/usr/bin/env python3

"""
Most of the datasets and pre-trained models we used are downloaded from the internet. However, compute nodes often
do not have access to the internet. This is a simple script to download and cache the datasets and pre-trained
models used in this project.
"""

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DATASETS = ["multi_news", "multi_x_science_sum", "allenai/mslr2022", "ccdv/WCEP-10"]
MODELS = [
    "google/pegasus-multi_news",
    "allenai/PRIMERA-multinews",
    "allenai/PRIMERA-multixscience",
    "allenai/PRIMERA-wcep",
    "allenai/led-base-16384-ms2",
]

for model in tqdm(MODELS, desc="Downloading models"):
    _ = AutoTokenizer.from_pretrained(model, force_download=True)
    _ = AutoModel.from_pretrained(model, force_download=True)

for dataset in tqdm(DATASETS, desc="Downloading datasets"):
    # Some datasets have configs. Handle those separately.
    if dataset == "allenai/mslr2022":
        _ = load_dataset(dataset, "ms2", download_mode="force_redownload")
        _ = load_dataset(dataset, "cochrane", download_mode="force_redownload")
    else:
        _ = load_dataset(dataset, download_mode="force_redownload")
