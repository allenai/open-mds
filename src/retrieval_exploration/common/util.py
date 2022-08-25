import json
import re
import sys
import warnings
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flatten_dict
import numpy as np
import pandas as pd
from nltk.tokenize import wordpunct_tokenize
from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from tqdm import tqdm
from transformers import PreTrainedTokenizer

# Local constants
_BASELINE_DIR = "baseline"
_PERTURBATIONS_DIR = "perturbations"

_RESULTS_FILENAME = "all_results.json"
_TRAINER_STATE_FILENAME = "trainer_state.json"
_LOG_HISTORY_KEY = "log_history"

# Public constants
DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multi_news": "|||||"}
CACHE_DIR = user_cache_dir("retrieval-exploration", "ai2")


def sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def unflatten(iterable, lengths):
    unflattened = []
    for i in range(len(lengths)):
        start = lengths[i - 1] if i >= 1 else 0
        end = start + lengths[i]
        unflattened.append(iterable[start:end])
    return unflattened


def parse_omega_conf() -> Dict[str, Any]:
    # Assume anything that ends in ".yml" is a YAML file to be parsed, and everything else is cli args
    cli_args = [arg for arg in sys.argv[1:][:] if not arg.endswith(".yml")]
    yml_confs = [OmegaConf.load(arg) for arg in sys.argv[1:][:] if arg.endswith(".yml")]
    cli_conf = OmegaConf.from_dotlist(cli_args)
    # Merge the YAML configs in the order they were given, with the cli args taking precedence
    conf = OmegaConf.merge(*yml_confs, cli_conf)
    # HuggingFace expects a vanilla python dict, so perform the conversion here
    return OmegaConf.to_object(conf)


def split_docs(text: str, doc_sep_token: str) -> List[str]:
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns a list of each individual documents.
    """
    # It's possible to have one or more doc_sep_token at the very end of the string.
    # Strip them here so that we get the correct number of documents when we split on doc_sep_token.
    text = text.rstrip().removesuffix(doc_sep_token).rstrip()
    while text.endswith(doc_sep_token):
        text = text.removesuffix(doc_sep_token).rstrip()
    docs = [doc.strip() for doc in text.split(doc_sep_token)]
    return docs


def get_num_docs(text: str, doc_sep_token: str) -> int:
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns the number of individual documents.
    """
    # See: https://stackoverflow.com/a/3393470
    return len(list(filter(bool, split_docs(text, doc_sep_token=doc_sep_token))))


def get_doc_sep_token(tokenizer: PreTrainedTokenizer) -> str:
    """Returns a suitable document seperator token depending on `tokenizer`. In general, the
    function checks if this `tokenizer.name_or_path` has a special document token (defined in
    `common.util.DOC_SEP_TOKENS`). If that is not found, it then checks for: `tokenizer.sep_token`,
    `tokenizer.bos_token`, `tokenizer.eos_token` in that order. If these are all `None`, a
    `ValueError` is raised.
    """
    # PRIMERA models have their own special token, <doc-sep>.
    if "primera" in tokenizer.name_or_path.lower():
        return DOC_SEP_TOKENS["primera"]
    elif tokenizer.sep_token is not None:
        return tokenizer.sep_token
    elif tokenizer.bos_token is not None:
        return tokenizer.bos_token
    elif tokenizer.eos_token is not None:
        return tokenizer.eos_token
    else:
        raise ValueError(f"Could not determine a suitable document sperator token '{tokenizer.name_or_path}'")


def get_global_attention_mask(input_ids: List[List[int]], token_ids: List[int]) -> List[List[int]]:
    """Returns a corresponding global attention mask for `input_ids`, which is 1 for any tokens in
    `token_ids` (indicating the model should attend to those tokens) and 0 elsewhere (indicating the
    model should not attend to those tokens).

    # Parameters

    input_ids : `List[List[str]]`
        The input ids that will be provided to a model during the forward pass.
    token_ids : `List[List[str]]`
        The token ids that should be globally attended to.
    """

    # TODO (John): Ideally this would be vectorized
    global_attention_mask = [[1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids]
    return global_attention_mask


def truncate_multi_doc(
    text: str,
    doc_sep_token: str,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    num_docs: Optional[int] = None,
) -> str:
    """Given some `text`, which is assumed to be multiple documents joined by `doc_sep_token`,
    truncates each document (using `tokenizer`) so that the length of the concatenation of all
    documents does not exceed max_length. See https://aclanthology.org/2021.naacl-main.380/ and
    https://arxiv.org/abs/2110.08499 for more details. If `num_docs` is provided, the truncation
    is done as if there are `num_docs` number of input documents. This is useful to control
    for truncation when applying pertubations (e.g. additiion and deletion).
    """
    input_docs = split_docs(text, doc_sep_token=doc_sep_token)
    # If num_docs is not provided, determine it from the input text
    num_docs = num_docs or get_num_docs(text, doc_sep_token=doc_sep_token)
    # -2 to make room for the special tokens, -(len(docs) - 1) to make room for the doc sep tokens.
    max_doc_length = (max_length - 2 - (num_docs - 1)) // num_docs
    # Truncate each doc to its maximum allowed length
    truncated_docs = [
        # Going to join everything on a space at the end, so strip it off here.
        tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc, max_length=max_doc_length, truncation=True)).strip()
        for doc in input_docs
    ]
    return f" {doc_sep_token} ".join(truncated_docs)


def batch_decode_multi_doc(sequences, tokenizer: PreTrainedTokenizer, doc_sep_token: str, **kwargs):
    """Performs a similar function to HuggingFace Tokenizers batch_decode without removing the `doc_sep_token`."""

    # skip_special_tokens must be false, so if use provided it as true via kwargs, warn them.
    skip_special_tokens = kwargs.pop("skip_special_tokens", None)
    if skip_special_tokens:
        warnings.warn("`skip_special_tokens=True` was provided to batch_decode_multi_doc but will be ignored.")

    decoded_sequences = tokenizer.batch_decode(sequences, skip_special_tokens=False, **kwargs)
    pattern = rf"{tokenizer.pad_token}"
    if tokenizer.bos_token is not None and tokenizer.bos_token != doc_sep_token:
        pattern += rf"|{tokenizer.bos_token}"
    if tokenizer.eos_token is not None and tokenizer.eos_token != doc_sep_token:
        pattern += rf"|{tokenizer.eos_token}"
    decoded_sequences = [re.sub(pattern, "", inputs).strip() for inputs in decoded_sequences]
    return decoded_sequences


def preprocess_multi_news(text: str, summary: str, doc_sep_token: str) -> Tuple[str, str]:
    text = text.strip(DOC_SEP_TOKENS["multi_news"]).strip()
    text = text.replace(DOC_SEP_TOKENS["multi_news"], doc_sep_token)
    summary = summary.strip()
    return text, summary


def preprocess_multi_x_science_sum(
    text: str, summary: str, ref_abstract: Dict[str, List[str]], doc_sep_token: str
) -> Tuple[str, str]:
    # Multi-XScience has empty abstracts. Drop them to avoid problems downstream.
    abstracts = [abstract.strip() for abstract in ref_abstract["abstract"] if abstract.strip()]
    text = f" {doc_sep_token} ".join([text.strip()] + abstracts)
    summary = summary.strip()
    return text, summary


def preprocess_ms2(
    text: str,
    summary: str,
    titles: List[str],
    abstracts: List[str],
    doc_sep_token: str,
    max_included_studies: int = 25,
) -> Tuple[str, str]:
    background = text.strip()
    articles = [f"{title.strip()} {abstract.strip()}" for title, abstract in zip(titles, abstracts)]
    # Following https://arxiv.org/abs/2104.06486, take the first 25 articles.
    articles = articles[:max_included_studies]
    text = f" {doc_sep_token} ".join([background] + articles)
    summary = summary.strip()
    return text, summary


def jaccard_similarity_score(string_1: str, string_2: str) -> float:
    """Returns the Jaccard similarity score between two strings, by comparing their token sets. Returns 1.0
    if both strings are empty."""
    string_1 = sanitize_text(string_1, lowercase=True)
    string_2 = sanitize_text(string_2, lowercase=True)
    string_1_tokens = set(wordpunct_tokenize(string_1.strip()))
    string_2_tokens = set(wordpunct_tokenize(string_2.strip()))
    if not string_1_tokens and not string_2_tokens:
        warnings.warn("Both string_1 and string_2 are empty. Returning 1.0.")
        return 1.0
    return len(string_1_tokens & string_2_tokens) / len(string_1_tokens | string_2_tokens)


def fraction_docs_perturbed(pre_perturbation: str, post_perturbation: str, doc_sep_token: str) -> float:
    """Given two strings, `pre_perturbation` and `post_perturbation`, representing the documents
    (separated by `doc_sep_token`) of an example before and after perturbation, returns the fraction of documents
    that were perturbed.
    """
    pre_perturbation = sanitize_text(pre_perturbation, lowercase=True)
    post_perturbation = sanitize_text(post_perturbation, lowercase=True)

    if not pre_perturbation:
        warnings.warn("pre_perturbation string is empty. Returning 0.0.")
        return 0.0

    pre_perturbation_docs = split_docs(pre_perturbation, doc_sep_token=doc_sep_token)
    post_perturbation_docs = split_docs(post_perturbation, doc_sep_token=doc_sep_token)

    num_perturbed = 0

    # If there are less documents post-perturbation, this is deletion, and we need to compute things differently
    if len(post_perturbation_docs) < len(pre_perturbation_docs):
        num_perturbed = len([doc for doc in pre_perturbation_docs if doc not in post_perturbation_docs])
    else:
        for pre, post in zip_longest(pre_perturbation_docs, post_perturbation_docs):
            if pre is None or post is None or pre != post:
                num_perturbed += 1
    return num_perturbed / len(pre_perturbation_docs)


def _read_result_dict(results_dict: Union[Dict[str, Any], List[Dict[str, Any]]], **kwargs) -> pd.DataFrame:
    """Reads an arbitrary dictionary or list of dictionaries, flattens it by joining all nested keys
    with a `'_'`, and returns it as a single pandas DataFrame. `**kwargs` are passed to `pd.DataFrame`.
    """
    if isinstance(results_dict, list):
        dfs = (pd.DataFrame(flatten_dict.flatten(rd, reducer="underscore"), **kwargs) for rd in results_dict)
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(flatten_dict.flatten(results_dict, reducer="underscore"), **kwargs)


def load_results_dicts(
    data_dir: str, metric_columns: Optional[List[str]] = None
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Loads the result dictionaries at `data_dir`. Assumes this directory is organized as follows:

    data_dir
    ├── model_1
    │   ├── baseline
    │   └── perturbations
    |       ├── perturbation_1
    |       |   └── all_results.json
    │       └── perturbation_2
    |           └── ...
    ├── model_2
    │   └── ...
    └── ...

    If the subdirectory `"baseline"` is present and `metric_columns` is provided, a new column
    `<metric>_delta`, one for each `metric` in `metric_columns`, will be computed as the difference
    between the perturbed results and the baseline results for the column `metric` and added to the
    returned dataframe.
    """

    baseline_dfs = []
    perturbation_dfs = []

    for model_dir in Path(data_dir).iterdir():
        baseline_df = None
        baseline_dir = Path(model_dir) / _BASELINE_DIR
        if baseline_dir.is_dir():
            # We need to parse different files depending if the model was trained or not.
            # These files require slightly different parsing strategies.
            if (baseline_dir / _TRAINER_STATE_FILENAME).is_file():
                filepath = baseline_dir / _TRAINER_STATE_FILENAME
                results_dict = json.loads(filepath.read_text())
                # TODO: This makes the assumption that logging happens ONLY at the end of training,
                # in which case all other elements in the log_history are evaluations. Load each
                # as a separate DataFrame and concatenate them.
                results_dict = results_dict[_LOG_HISTORY_KEY][:-1]
            elif (baseline_dir / _RESULTS_FILENAME).is_file():
                filepath = baseline_dir / _RESULTS_FILENAME
                results_dict = json.loads(filepath.read_text())

            else:
                raise ValueError(
                    f"Did not find any of the expected files in {baseline_dir}. Looking for one of"
                    f" {_RESULTS_FILENAME} or {_TRAINER_STATE_FILENAME}."
                )
            baseline_df = _read_result_dict(results_dict)
            baseline_dfs.append(baseline_df)

        perturbation_dir = Path(model_dir) / _PERTURBATIONS_DIR
        filepaths = list(Path(perturbation_dir).glob(f"**/{_TRAINER_STATE_FILENAME}"))
        if filepaths:
            train = True
        else:
            train = False
            filepaths = list(Path(perturbation_dir).glob(f"**/{_RESULTS_FILENAME}"))

        for filepath in tqdm(filepaths):
            results_dict = json.loads(filepath.read_text())
            if train:
                perturbation_df = _read_result_dict(results_dict[_LOG_HISTORY_KEY][:-1])
            else:
                perturbation_df = _read_result_dict(results_dict)
            if baseline_df is not None:
                # The perturbation and baseline data should pertain to the same examples.
                if not np.array_equal(baseline_df.predict_example_idx, perturbation_df.predict_example_idx):
                    raise ValueError("The perturbation and baseline data do not correspond to the same examples!")

                perturbation_df["jaccard_similarity_scores"] = [
                    jaccard_similarity_score(pre, post)
                    for pre, post in zip(perturbation_df.predict_inputs, baseline_df.predict_inputs)
                ]
                if metric_columns is not None:
                    for metric in metric_columns:
                        # Compute the per-instance absolute differences
                        perturbation_df[f"{metric}_delta"] = perturbation_df[metric] - baseline_df[metric]
            perturbation_dfs.append(perturbation_df)
    baseline_df = pd.concat(baseline_dfs, ignore_index=True) if baseline_dfs else None
    perturbed_df = pd.concat(perturbation_dfs, ignore_index=True)
    return baseline_df, perturbed_df
