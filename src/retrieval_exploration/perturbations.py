import math
import random
from typing import List, Optional

from retrieval_exploration.common import util


def _sample_random_docs(
    inputs: List[str],
    doc_sep_token: str,
    k: int,
    exclude: Optional[List[int]] = None,
    seed: Optional[int] = None,
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
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` to we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    # Easier to deal with an empty list than None, but bad practice to set default value as []
    exclude = exclude or []

    # Check that we have enough documents to sample from
    total_num_docs = sum(
        len(util.split_docs(input_, doc_sep_token=doc_sep_token))
        for i, input_ in enumerate(inputs)
        if i not in exclude
    )
    if total_num_docs < k:
        raise ValueError(
            f"Not enough documents to sample {k} without replacement. Only have {total_num_docs}."
        )

    random_docs = []
    while True:
        random_instance_idx = rng.randint(0, len(inputs) - 1)
        # Don't sample documents from the example we are currently processing
        if exclude is not None and random_instance_idx in exclude:
            continue
        random_example = inputs[random_instance_idx]
        docs = random_example.strip(doc_sep_token).strip().split(doc_sep_token)
        # Don't sample the same random document (for the same instance) twice
        while True:
            random_doc = rng.choice(docs)
            if random_doc not in random_docs:
                random_docs.append(random_doc)
                break
        if len(random_docs) == k:
            break
    return random_docs


def random_shuffle(
    inputs: List[str],
    doc_sep_token: str,
    per_perturbed: Optional[float] = None,
    seed: Optional[int] = None,
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
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` to we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    perturbed_inputs = []

    for example in inputs:
        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)
        rng.shuffle(input_docs)
        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs


def random_addition(
    inputs: List[str],
    doc_sep_token: str,
    per_perturbed: Optional[float] = None,
    seed: Optional[int] = None,
):
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
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for i, example in enumerate(inputs):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample k unique documents to add
        random_docs = _sample_random_docs(
            inputs, doc_sep_token=doc_sep_token, k=k, exclude=[i], seed=seed
        )

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + random_docs))

    return perturbed_inputs


def random_deletion(
    inputs: List[str],
    doc_sep_token: str,
    per_perturbed: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by removing `per_perturbed`
    percent of documents in each example at random.

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    per_perturbed : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` to we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for example in inputs:

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample k unique documents to delete
        to_delete = rng.sample(range(len(input_docs)), k)

        # Collect the perturbed example
        perturbed_inputs.append(
            f" {doc_sep_token} ".join(doc for j, doc in enumerate(input_docs) if j not in to_delete)
        )

    return perturbed_inputs


def random_duplication(
    inputs: List[str],
    doc_sep_token: str,
    per_perturbed: Optional[float] = None,
    seed: Optional[int] = None,
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
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` to we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for example in inputs:
        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to add
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample k documents (without replacement) which we will repeat in the input
        repeaters = rng.sample(input_docs, k)

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + repeaters))

    return perturbed_inputs


def random_replacement(
    inputs: List[str],
    doc_sep_token: str,
    per_perturbed: Optional[float] = None,
    seed: Optional[int] = None,
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
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` to we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    if not per_perturbed:
        return inputs

    perturbed_inputs = []

    for i, example in enumerate(inputs):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(per_perturbed * len(input_docs))

        # Randomly sample k unique documents
        random_docs = _sample_random_docs(
            inputs, doc_sep_token=doc_sep_token, k=k, exclude=[i], seed=seed
        )

        # Replace random documents in the current instance with the randomly choosen documents
        for j, doc in zip(rng.sample(range(len(input_docs)), k), random_docs):
            input_docs[j] = doc.strip()

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs
