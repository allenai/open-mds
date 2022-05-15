import math
import random
from typing import List, Optional

from retrieval_exploration.common import util


def _sample_random_docs(
    inputs: List[str], doc_sep_token: str, k: int, exclude: Optional[List[int]] = None
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, randomly samples `k` documents without
    replacement. If `exclude` is provided, will not sample from the examples at these indices in
    `inputs`.

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    k : `int`
        The number of documents to sample (without replacement) from `inputs`.
    exclude : `List[int]`, optional (default=None)
        If provided, will not sample from the examples at these indices in `inputs`.
    """
    random_docs = []
    while True:
        random_instance_idx = random.randint(0, len(inputs) - 1)
        # Don't sample documents from the example we are currently processing
        if exclude is not None and random_instance_idx in exclude:
            continue
        random_example = inputs[random_instance_idx]
        docs = random_example.strip(doc_sep_token).strip().split(doc_sep_token)
        # Don't sample the same random document (for the same instance) twice
        while True:
            random_doc = random.choice(docs)
            if random_doc not in random_docs:
                random_docs.append(random_doc)
                break
        if len(random_docs) == k:
            break
    return random_docs


def random_shuffle(
    inputs: List[str], doc_sep_token: str, per_perturbed: Optional[float] = None
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by randomly shuffling the
    order of documents in each example.

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`, optional (default=None)
        Has no effect. Exists for consistency with other perturbation functions.
    """
    perturbed_inputs = []

    for text in inputs:
        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)
        random.shuffle(input_docs)
        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs


def random_addition(inputs: List[str], doc_sep_token: str, per_perturbed: Optional[float] = None):
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by adding `per_perturbed`
    percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    """
    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for i, text in enumerate(inputs):

        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample documents k unique documents
        random_docs = _sample_random_docs(inputs, doc_sep_token=doc_sep_token, k=k, exclude=[i])

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + random_docs))

    return perturbed_inputs


def random_deletion(inputs: List[str], doc_sep_token: str, per_perturbed: Optional[float] = None):
    pass


def random_duplication(
    inputs: List[str], doc_sep_token: str, per_perturbed: Optional[float] = None
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by replacing `per_perturbed`
    percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    """
    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for text in inputs:
        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)

        # The absolute number of documents to add
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample k documents (without replacement) which we will repeat in the input
        repeaters = random.sample(input_docs, k)

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + repeaters))

    return perturbed_inputs


def random_replacement(
    inputs: List[str], doc_sep_token: str, per_perturbed: Optional[float] = None
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by replacing `per_perturbed`
    percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    """
    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for i, text in enumerate(inputs):

        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample documents k unique documents
        random_docs = _sample_random_docs(inputs, doc_sep_token=doc_sep_token, k=k, exclude=[i])

        # Replace random documents in the current instance with the randomly choosen documents
        for j, doc in zip(random.sample(range(len(input_docs)), k), random_docs):
            input_docs[j] = doc.strip()

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs
