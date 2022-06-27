import copy
import math
import random
from itertools import zip_longest
from typing import List, Optional, Tuple

import datasets
import nlpaug.augmenter.word as naw
import nltk
import torch

import more_itertools

from retrieval_exploration.common import util
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


def _randomly_sample_docs(
    inputs: List[str],
    doc_sep_token: str,
    k: Optional[int] = None,
    query: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, randomly samples `k` documents without
    replacement. Note that documents will NOT be sampled from the `query` example in `inputs`.

    # Parameters

    query : `str`
        Documents will NOT be sampled from this example in `inputs`.
    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    k : `int`
        The number of documents to sample (without replacement) from `inputs`.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    # Instantiate an instance of `Random` so we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    # If query is provided, remove it from the possible inputs
    if query is not None:
        inputs = copy.deepcopy(inputs)
        inputs = [example for example in inputs if example != query]

    # Sample all documents if k is not provided
    total_num_docs = sum(len(util.split_docs(example, doc_sep_token=doc_sep_token)) for example in inputs)
    k = k or total_num_docs
    # Check that we have enough documents to sample from
    if total_num_docs < k:
        raise ValueError(f"Not enough documents to sample {k} without replacement. Only have {total_num_docs}.")

    random_docs = []
    while True:
        random_instance_idx = rng.randint(0, len(inputs) - 1)
        random_example = inputs[random_instance_idx]
        docs = util.split_docs(random_example, doc_sep_token=doc_sep_token)
        # Check that we haven't already sampled all documents from this example
        if all(doc in random_docs for doc in docs):
            continue
        # Don't sample the same random document (for the same instance) twice
        while True:
            random_doc = rng.choice(docs)
            if random_doc not in random_docs:
                random_docs.append(random_doc)
                break
        if len(random_docs) == k:
            break
    return random_docs


def _semantically_sample_docs(
    inputs: List[str],
    doc_sep_token: str,
    strategy: str,
    k: Optional[int] = None,
    query: Optional[str] = None,
    target: Optional[str] = None,
):
    if strategy not in ["similar", "dissimilar"]:
        raise ValueError(f"Got unknown sampling strategy: {strategy}. Expected one of {['similar', 'dissimilar']}")
    if not query and not target:
        raise ValueError("Must provide either a `query` or a `target`.")

    # If query is provided, remove it from the possible inputs
    query_docs = []
    if query is not None:
        inputs = copy.deepcopy(inputs)
        inputs = [example for example in inputs if example != query]
        query_docs = util.split_docs(query, doc_sep_token=doc_sep_token)

    # Sample all documents if k is not provided
    total_num_docs = sum(len(util.split_docs(example, doc_sep_token=doc_sep_token)) for example in inputs)
    k = k or total_num_docs
    # Check that we have enough documents to sample from
    if total_num_docs < k:
        raise ValueError(f"Not enough documents to sample {k} without replacement. Only have {total_num_docs}.")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    scored_docs: List[Tuple[str, float]] = []
    for example in inputs:
        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)
        input_docs_embeddings = embedder.encode(input_docs, convert_to_tensor=True)

        # If a target is provided, we look for lexical overlap between it and the input documents.
        # Otherwise we use the documents of the query.
        if target:
            query_embedding = embedder.encode(target, convert_to_tensor=True)
            scores = cos_sim(query_embedding, input_docs_embeddings)[0]

        else:
            query_embedding = embedder.encode(query_docs, convert_to_tensor=True)
            scores = cos_sim(query_embedding, input_docs_embeddings)
            scores = torch.mean(scores, axis=0)

        scored_docs.extend(zip(input_docs, scores.tolist()))

    # Return the the top k most similar (or dissimilar) documents
    scored_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True if strategy == "similar" else False,
    )
    scored_docs = scored_docs[:k]
    return [doc for doc, _ in scored_docs]


def sorting(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
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
    perturbed_frac : `float`, optional (default=None)
        Has no effect. Exists for consistency with other perturbation functions.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    # Instantiate an instance of `Random` so we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        if strategy == "random":
            rng.shuffle(input_docs)
        else:
            input_docs = _semantically_sample_docs(
                inputs=[example],
                doc_sep_token=doc_sep_token,
                strategy=strategy,
                target=target,
            )

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs


def addition(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = None,
):
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by adding `perturbed_frac`
    percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    perturbed_frac : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # No-op if perturbed_frac is None or falsey
    if not perturbed_frac:
        return inputs

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * len(input_docs))

        if strategy == "random":
            sampled_docs = _randomly_sample_docs(
                inputs=inputs, doc_sep_token=doc_sep_token, k=k, query=example, seed=seed
            )
        else:
            sampled_docs = _semantically_sample_docs(
                inputs=inputs,
                doc_sep_token=doc_sep_token,
                k=k,
                strategy=strategy,
                query=example,
                target=target,
            )

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + sampled_docs))

    return perturbed_inputs


def deletion(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = None,
):
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by removing `perturbed_frac`
    percent of documents in each example at random.

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    perturbed_frac : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # No-op if perturbed_frac is None or falsey
    if not perturbed_frac:
        return inputs

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    # Instantiate an instance of `Random` so we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * len(input_docs))

        if strategy == "random":
            to_delete = rng.sample(range(len(input_docs)), k)
        else:
            sampled_docs = _semantically_sample_docs(
                inputs=[example],
                doc_sep_token=doc_sep_token,
                k=k,
                strategy=strategy,
                target=target,
            )
            to_delete = [input_docs.index(doc) for doc in sampled_docs]

        # Collect the perturbed example
        perturbed_inputs.append(
            f" {doc_sep_token} ".join(doc for j, doc in enumerate(input_docs) if j not in to_delete)
        )

    return perturbed_inputs


def duplication(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = None,
) -> List[str]:
    """Given `inputs`, a list of strings where each string contains the input documents seperated
    by `doc_sep_token` of one example from the dataset, perturbs the input by replacing `perturbed_frac`
    percent of documents in each example with a random document sampled from `inputs.`

    # Parameters

    inputs : `List[str]`
        A list of strings, each string containing the input documents for one example. It is assumed
        that documents are seperated by `doc_sep_token`.
    doc_sep_token : `str`
        The token that separates individual documents in `inputs`.
    perturbed_frac : `float`, optional (default=None)
        The percentage of documents in each example that should be randomly replaced with a document
        sampled from `inputs`. If None (or falsey), no documents will be perturbed as this is a no-op.
    seed : `int`, optional (default=None)
        If provided, will locally set the seed of the `random` module with this value.
    """
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # No-op if perturbed_frac is None or falsey
    if not perturbed_frac:
        return inputs

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    # Instantiate an instance of `Random` so we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * len(input_docs))

        if strategy == "random":
            repeaters = rng.sample(input_docs, k)
        else:
            repeaters = _semantically_sample_docs(
                inputs=[example],
                doc_sep_token=doc_sep_token,
                k=k,
                strategy=strategy,
                target=target,
            )

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs + repeaters))

    return perturbed_inputs


def replacement(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = None,
):
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # No-op if perturbed_frac is None or falsey
    if not perturbed_frac:
        return inputs

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * len(input_docs))

        if strategy == "random":
            sampled_docs = _randomly_sample_docs(
                inputs=inputs, doc_sep_token=doc_sep_token, k=k, query=example, seed=seed
            )
        else:
            sampled_docs = _semantically_sample_docs(
                inputs=inputs,
                doc_sep_token=doc_sep_token,
                k=k,
                strategy=strategy,
                query=example,
                target=target,
            )

        for i, doc in zip(random.sample(range(len(input_docs)), k), sampled_docs):
            input_docs[i] = doc.strip()

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs


def backtranslation(
    inputs: List[str],
    doc_sep_token: str,
    targets: Optional[List[str]] = None,
    perturbed_frac: Optional[float] = None,
    strategy: str = "random",
    seed: Optional[int] = None,
) -> List[str]:
    if strategy not in ["random", "similar", "dissimilar"]:
        raise ValueError(
            (f"Got unknown sampling strategy: {strategy}. Expected one of {['random', 'similar', 'dissimilar']}")
        )

    # No-op if perturbed_frac is None or falsey
    if not perturbed_frac:
        return inputs

    # Need an iterable, but an empty list as default value is bad practice
    targets = targets or []

    # Instantiate an instance of `Random` so we can create a generator with its own local seed
    # See: https://stackoverflow.com/a/37356024/6578628
    rng = random.Random(seed)

    # Load the back-translation augmenter
    device = "cuda" if torch.cuda.is_available() else "cpu"
    aug = naw.BackTranslationAug(
        from_model_name="Helsinki-NLP/opus-mt-en-da",
        to_model_name="Helsinki-NLP/opus-mt-da-en",
        device=device,
        max_length=256,
    )

    perturbed_inputs = []
    for example, target in zip_longest(inputs, targets):

        input_docs = util.split_docs(example, doc_sep_token=doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * len(input_docs))

        if strategy == "random":
            sampled_docs = rng.sample(input_docs, k)
        else:
            sampled_docs = _semantically_sample_docs(
                inputs=[example],
                doc_sep_token=doc_sep_token,
                k=k,
                strategy=strategy,
                target=target,
            )

        # Back translate the sampled documents. To take advantage of batching, we will
        # collect the sentences of all documents, pass them to the model, and then unflatten them.
        unflattened_sents = [nltk.sent_tokenize(doc) for doc in sampled_docs]
        back_translated_sents = aug.augment(list(more_itertools.flatten(unflattened_sents)))
        back_translated_docs = util.unflatten(
            back_translated_sents, lengths=[len(sents) for sents in unflattened_sents]
        )

        for sampled, translated in zip(sampled_docs, back_translated_docs):
            input_docs[input_docs.index(sampled)] = " ".join(sent.strip() for sent in translated)

        perturbed_inputs.append(f" {doc_sep_token} ".join(input_docs))

    return perturbed_inputs
