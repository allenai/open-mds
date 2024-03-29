import json
import re
import sys
import warnings
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import flatten_dict
import pandas as pd
import tiktoken
from nltk.tokenize import wordpunct_tokenize
from omegaconf import OmegaConf
from platformdirs import user_cache_dir
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# Local constants
_BASELINE_DIR = "baseline"
_PERTURBATIONS_DIR = "perturbations"
_RETRIEVAL_DIR = "retrieval"
_TRAINING_DIR = "training"
_RESULTS_FILENAME = "all_results.json"

# Public constants
DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multi_news": "|||||", "ccdv/WCEP-10": "</s>"}
CACHE_DIR = user_cache_dir("open-mds", "ai2")


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


def get_doc_sep_token(tokenizer: PreTrainedTokenizerBase) -> str:
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
    tokenizer: Union[PreTrainedTokenizerBase, tiktoken.core.Encoding],
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

    # Setup encode/decode functions based on the type of tokenizer
    if isinstance(tokenizer, PreTrainedTokenizerBase):
        # make room for the start/end special tokens
        max_length -= 2
        encode = tokenizer.tokenize
        decode = tokenizer.convert_tokens_to_string
    elif isinstance(tokenizer, tiktoken.core.Encoding):
        encode = tokenizer.encode
        decode = tokenizer.decode
    else:
        raise ValueError(
            f"tokenizer must be either a PreTrainedTokenizerBase or a tiktoken.core.Encoding, got {type(tokenizer)}"
        )
    if len(encode(text)) > max_length:
        # If num_docs is not provided, determine it from the input text
        num_docs = num_docs or get_num_docs(text, doc_sep_token=doc_sep_token)
        # make room for doc_sep_token's
        max_doc_length = max_length - len(encode(f" {doc_sep_token} ")) * (num_docs - 1)
        max_doc_length = max_doc_length // num_docs
        # Going to join everything on a space at the end, so strip it off here.
        truncated_docs = [decode(encode(doc)[:max_doc_length]).strip() for doc in input_docs]
    else:
        truncated_docs = input_docs
    return f" {doc_sep_token} ".join(truncated_docs)


def batch_decode_multi_doc(sequences, tokenizer: PreTrainedTokenizerBase, doc_sep_token: str, **kwargs):
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


def preprocess_wcep(text: str, summary: str, doc_sep_token: str) -> Tuple[str, str]:
    text = text.strip(DOC_SEP_TOKENS["ccdv/WCEP-10"]).strip()
    text = text.replace(DOC_SEP_TOKENS["ccdv/WCEP-10"], doc_sep_token)
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
    # Following https://aclanthology.org/2021.emnlp-main.594/, take the first 25 articles.
    articles = articles[:max_included_studies]
    text = f" {doc_sep_token} ".join([background] + articles)
    summary = summary.strip()
    return text, summary


def preprocess_cochrane(
    summary: str, titles: List[str], abstracts: List[str], doc_sep_token: str, max_included_studies: int = 25
) -> Tuple[str, str]:
    articles = [f"{title.strip()} {abstract.strip()}" for title, abstract in zip(titles, abstracts)]
    # Following https://aclanthology.org/2021.emnlp-main.594/, take the first 25 articles.
    articles = articles[:max_included_studies]
    text = f" {doc_sep_token} ".join(articles)
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


def get_frac_docs_perturbed(pre_perturbation: str, post_perturbation: str, doc_sep_token: str) -> float:
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


def get_pyterrier_versions() -> Tuple[str, str]:
    """Returns the versions of the currently installed Terrier assembly and Terrier python helper jars. These
    are required to use PyTerrier offline.
    """
    # Get the filenames of the terrier assembly and terrier python helper jars
    pyterrier_path = Path.home() / ".pyterrier"
    terrier_assemblies_fn = list(pyterrier_path.glob("terrier-assemblies-*.jar"))[0].stem
    terrier_python_helper_fn = list(pyterrier_path.glob("terrier-python-helper-*.jar"))[0].stem
    # Extract the versions from each
    version = terrier_assemblies_fn.lstrip("terrier-assemblies-").split("-")[0]
    helper_version = terrier_python_helper_fn.lstrip("terrier-python-helper-")
    return version, helper_version


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
    data_dir: str,
    include_models: Optional[List[str]] = None,
    metric_columns: Optional[List[str]] = None,
    metric_key_prefix: Optional[str] = "predict",
    load_perturbation_results: bool = True,
    load_retrieval_results: bool = True,
    load_training_results: bool = True,
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """Loads the result dictionaries at `data_dir`. Assumes this directory is organized as follows:

    data_dir
    ├── model_1
    │   ├── baseline
    │   ├── perturbations
    |   |   ├── perturbation_1
    |   |   |   └── all_results.json
    │   |   └── perturbation_2
    |   |       └── ...
    │   ├── retrieval
    │   |   └── ...
    │   └── training
    │       └── ...
    ├── model_2
    │   └── ...
    └── ...

    If the subdirectory `"baseline"` is present and `metric_columns` is provided, a new column `<metric>_delta`,
    one for each `metric` in `metric_columns`, will be computed as the difference between the experimental results
    and the baseline results for the column `metric` and added to the returned dataframe. The metrics are assumed
    to be prefixed with `metric_key_prefix`, which is added by the HuggingFace Trainer (and defaults to "eval" for
    the validation set and "predict" for the test set). You can control which sets of results are loaded with
    `load_perturbation_results`, `load_retrieval_results`, and `load_training_results`. To only load the results
    for some models, provide a list of model names, `include_models`.
    """

    baseline_dfs = []
    results_dfs = []

    for model_dir in Path(data_dir).iterdir():

        # Only load data from the specified model directories, if provided
        if include_models is not None and model_dir.name not in include_models:
            continue

        baseline_df = None
        baseline_dir = Path(model_dir) / _BASELINE_DIR

        # Load baseline data, if it exists
        if baseline_dir.is_dir():
            filepath = baseline_dir / _RESULTS_FILENAME
            results_dict = json.loads(filepath.read_text())
            baseline_df = _read_result_dict(results_dict)
            baseline_dfs.append(baseline_df)

        # Load the experimental results
        filepaths = []
        if load_perturbation_results:
            perturbation_dir = Path(model_dir) / _PERTURBATIONS_DIR
            filepaths += list(Path(perturbation_dir).glob(f"**/{_RESULTS_FILENAME}"))

        if load_retrieval_results:
            retrieval_dir = Path(model_dir) / _RETRIEVAL_DIR
            filepaths += list(Path(retrieval_dir).glob(f"**/{_RESULTS_FILENAME}"))

        if load_training_results:
            training_dir = Path(model_dir) / _TRAINING_DIR
            filepaths += list(Path(training_dir).glob(f"**/{_RESULTS_FILENAME}"))

        # Process the individual results files and store them as dataframes
        for filepath in tqdm(filepaths, desc=f"Loading results from {model_dir}"):
            results_dict = json.loads(filepath.read_text())
            results_df = _read_result_dict(results_dict)

            # This is a little brittle, but if the filepath is named after a checkpoint, save it in the results_df
            checkpoint = re.search(r"checkpoint-(\d+)", str(filepath.parent.name))
            results_df["checkpoint"] = int(checkpoint.group(1)) if checkpoint is not None else None

            if baseline_df is not None:
                # The perturbation and baseline data should pertain to the same examples.
                for i, (baseline_ref_summ, results_ref_summ) in enumerate(
                    zip(baseline_df[f"{metric_key_prefix}_labels"], results_df[f"{metric_key_prefix}_labels"])
                ):
                    # Sanitize because we don't care about minor differences like whitespace (handled during eval).
                    # if sanitize_text(baseline_ref_summ) != sanitize_text(results_ref_summ):
                    #     raise ValueError(
                    #         f"For at least one example (index={i}), the baseline and experimental results"
                    #         f" dataframes have different reference summaries. This may indicate that baseline and"
                    #         " experimental results do not correspond to the same examples."
                    #     )
                    pass

                # Compute the actual number of documents perturbed
                results_df["frac_docs_perturbed"] = [
                    get_frac_docs_perturbed(pre, post, doc_sep_token=doc_sep_token)
                    for pre, post, doc_sep_token in zip(
                        baseline_df[f"{metric_key_prefix}_inputs"],
                        results_df[f"{metric_key_prefix}_inputs"],
                        baseline_df.doc_sep_token,
                    )
                ]

                # Compute the absolute difference between the baseline and experimental results for these metrics
                if metric_columns is not None:
                    for metric in metric_columns:
                        # Compute the per-instance absolute differences
                        results_df[f"{metric}_delta"] = results_df[metric] - baseline_df[metric]

            results_dfs.append(results_df)

    if not results_dfs:
        raise ValueError(f"No experimental results were found in {data_dir}.")

    baseline_df = pd.concat(baseline_dfs, ignore_index=True) if baseline_dfs else None
    results_df = pd.concat(results_dfs, ignore_index=True)
    return baseline_df, results_df
