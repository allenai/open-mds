import math
import random
from typing import List, Tuple

from transformers import PreTrainedTokenizer

_DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multi_news": "|||||"}


def preprocess_multi_news(text: str, summary: str, doc_sep_token: str) -> Tuple[str, str]:
    # Some examples have a doc sep token at the end of the text, so strip it before we split.
    text = text.strip(_DOC_SEP_TOKENS["multi_news"]).strip()
    text = text.replace(_DOC_SEP_TOKENS["multi_news"], doc_sep_token)
    return text, summary


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
        raise ValueError(
            f"Could not determine a suitable document sperator token '{tokenizer.name_or_path}'"
        )


def truncate_multi_doc(
    text: str, doc_sep_token: str, max_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    """Given some `text`, which is assumed to be multiple documents joined by `doc_sep_token`,
    truncates each document (using `tokenizer`) so that the length of the concatenation of all
    documents does not exceed max_length. See https://aclanthology.org/2021.naacl-main.380/ and
    https://arxiv.org/abs/2110.08499 for more details.
    """
    # Some datasets have the doc sep token at the end of the text, so strip it before we split.
    docs = text.strip(doc_sep_token).strip().split(doc_sep_token)
    # -2 to make room for the special tokens, -(len(docs) - 1) to make room for the doc sep tokens.
    max_doc_length = (max_length - 2 - (len(docs) - 1)) // len(docs)
    truncated_docs = []
    for doc in docs:
        # Truncate each doc to its maximum allowed length
        truncated_docs.append(
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(doc, max_length=max_doc_length, truncation=True)
                # Going to join everything on a space at the end, so strip it off here.
            ).strip()
        )
    return f" {doc_sep_token} ".join(truncated_docs)


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
    global_attention_mask = [
        [1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids
    ]
    return global_attention_mask


def perturb(inputs: List[str], doc_sep_token: str, per_perturbed: float) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents
    (seperated `doc_sep_token`) of one example from the dataset, perturbs the input by replacing
    `per_perturbed` percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`.
    """
    # Do nothing if no documents should be perturbed
    if per_perturbed == 0.0:
        return inputs

    perturbed_inputs = []

    for i, text in enumerate(inputs):
        # Some datasets have the doc sep token at the end of the text, so strip it before we split.
        # We also strip off extra whitespace at the beginning and end of each document because we
        # will join on a space later.
        input_docs = [doc.strip() for doc in text.strip(doc_sep_token).strip().split(doc_sep_token)]

        # The absolute number of documents to replace
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample documents until we have at least k of them
        random_docs = []
        while True:
            random_instance_idx = random.randint(0, len(inputs) - 1)
            # Don't sample documents from the example we are currently processing
            if random_instance_idx == i:
                continue
            random_example = inputs[random_instance_idx]
            docs = random_example.strip(doc_sep_token).strip().split(doc_sep_token)
            # Don't sample the same random document (for the same instance) twice
            while True:
                random_doc = random.choice(docs)
                if random_doc not in random_docs:
                    break
            random_docs.append(random_doc)
            if len(random_docs) == k:
                break

        # Replace random documents in the current instance with the randomly choosen documents
        for j, doc in zip(random.sample(range(len(input_docs)), k), random_docs):
            input_docs[j] = doc.strip()

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs
