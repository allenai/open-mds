import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flatten_dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PretrainedConfig, PreTrainedTokenizer

# Local constants
_DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multi_news": "|||||"}

_BASELINE_DIR = "baseline"
_PERTURBATIONS_DIR = "perturbations"

_RESULTS_FILENAME = "all_results.json"
_TRAINER_STATE_FILENAME = "trainer_state.json"
_LOG_HISTORY_KEY = "log_history"


def unflatten(iterable, lengths):
    unflattened = []
    for i in range(len(lengths)):
        start = lengths[i - 1] if i >= 1 else 0
        end = start + lengths[i]
        unflattened.append(iterable[start:end])
    return unflattened


def jaccard_similarity_score(string_1: str, string_2: str) -> float:
    """Returns the Jaccard similarity score between two strings, using the sets of whitespace tokens. Returns 1.0
    if both strings are empty."""
    string_1_tokens = set(string_1.strip().split())
    string_2_tokens = set(string_2.strip().split())
    if not string_1_tokens and not string_2_tokens:
        warnings.warn("Both string_1 and string_2 are empty. Returning 1.0.")
        return 1.0
    return len(string_1_tokens & string_2_tokens) / len(string_1_tokens | string_2_tokens)


def split_docs(text: str, doc_sep_token: str) -> List[str]:
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns a list of each individual documents. Ignores any documents that are empty.
    order of documents in each example.
    """
    # It's possible to have a doc_sep_token at the very end of the string. Strip it here
    # so that we get the correct number of documents when we split on doc_sep_token.
    text = re.sub(rf"{doc_sep_token}$", "", text.strip())
    return [doc.strip() for doc in text.split(doc_sep_token)]


def get_num_docs(text: str, doc_sep_token: str) -> int:
    """Given `text`, a string which contains the input documents seperated by `doc_sep_token`,
    returns the number of individual documents.
    """
    # See: https://stackoverflow.com/a/3393470
    return len(list(filter(bool, split_docs(text, doc_sep_token=doc_sep_token))))


def get_doc_sep_token(tokenizer: PreTrainedTokenizer) -> str:
    """Returns a suitable document seperator token depending on `tokenizer`. In general, the
    function checks if this `tokenizer.name_or_path` has a special document token (defined in
    `common.util._DOC_SEP_TOKENS`). If that is not found, it then checks for: `tokenizer.sep_token`,
    `tokenizer.bos_token`, `tokenizer.eos_token` in that order. If these are all `None`, a
    `ValueError` is raised.
    """
    # PRIMERA models have their own special token, <doc-sep>.
    if "primera" in tokenizer.name_or_path.lower():
        return _DOC_SEP_TOKENS["primera"]
    elif tokenizer.sep_token is not None:
        return tokenizer.sep_token
    elif tokenizer.bos_token is not None:
        return tokenizer.bos_token
    elif tokenizer.eos_token is not None:
        return tokenizer.eos_token
    else:
        raise ValueError(f"Could not determine a suitable document sperator token '{tokenizer.name_or_path}'")


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


def preprocess_multi_news(text: str, summary: str, doc_sep_token: str) -> Tuple[str, str]:
    """Given an `example` dict, returns a tuple of strings containing the text and summary."""
    text = text.strip(_DOC_SEP_TOKENS["multi_news"]).strip()
    text = text.replace(_DOC_SEP_TOKENS["multi_news"], doc_sep_token)
    summary = summary.strip()
    return text, summary


def preprocess_multi_x_science_sum(
    text: str, summary: str, ref_abstract: Dict[str, List[str]], doc_sep_token: str
) -> Tuple[str, str]:
    """Given an `example` dict, returns a tuple of strings containing the text and summary."""
    abstracts = [abstract.strip() for abstract in ref_abstract["abstract"]]
    text = f" {doc_sep_token} ".join([text.strip()] + abstracts)
    summary = summary.strip()
    return text, summary


def get_task_specific_params(config: PretrainedConfig, task: str) -> Optional[Dict[str, Any]]:
    task_specific_params = None
    if config.task_specific_params is not None:
        task_specific_params = config.task_specific_params.get(task)
        if task_specific_params:
            config.update(task_specific_params)
    return task_specific_params


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
                if not np.array_equal(baseline_df.eval_example_idx, perturbation_df.eval_example_idx):
                    raise ValueError("The perturbation and baseline data do not correspond to same examples!")

                if metric_columns is not None:
                    for metric in metric_columns:
                        # Compute the aggregated, relative difference as a percent
                        perturbation_df[f"{metric}_rel_diff"] = (
                            (perturbation_df[metric].mean() - baseline_df[metric].mean())
                            / np.abs(baseline_df[metric].mean())
                        ) * 100
                        # Compute the per-instance absolute differences
                        perturbation_df[f"{metric}_delta"] = perturbation_df[metric] - baseline_df[metric]
            perturbation_dfs.append(perturbation_df)
    baseline_df = pd.concat(baseline_dfs, ignore_index=True) if baseline_dfs else None
    perturbed_df = pd.concat(perturbation_dfs, ignore_index=True)
    return baseline_df, perturbed_df
