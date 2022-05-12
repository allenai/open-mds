import math
import random
from typing import List

from retrieval_exploration.common import util


def random_replacement(inputs: List[str], doc_sep_token: str, per_perturbed: float) -> List[str]:
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
        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)

        # The absolute number of documents to replace
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample documents until we have at least k unique ones
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
                    random_docs.append(random_doc)
                    break
            if len(random_docs) == k:
                break

        # Replace random documents in the current instance with the randomly choosen documents
        for j, doc in zip(random.sample(range(len(input_docs)), k), random_docs):
            input_docs[j] = doc.strip()

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs


def random_shuffle(inputs: List[str], doc_sep_token: str) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents
    (seperated `doc_sep_token`) of one example from the dataset, perturbs the input by randomly
    shuffling the order of documents in each example.

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    """
    perturbed_inputs = []

    for text in inputs:
        # Some datasets have the doc sep token at the end of the text, so strip it before we split.
        # We also strip off extra whitespace at the beginning and end of each document because we
        # will join on a space later.
        input_docs = util.split_docs(text, doc_sep_token=doc_sep_token)
        random.shuffle(input_docs)
        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs
