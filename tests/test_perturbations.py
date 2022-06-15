import copy
import random

import math
import requests

from retrieval_exploration import perturbations
from retrieval_exploration.common import util
import pytest
from datasets import load_dataset


def test_randomly_sample_docs() -> None:
    num_docs = 8
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]

    # Choose a k greater than the total number of documents WITH a query
    with pytest.raises(ValueError):
        _ = perturbations._randomly_sample_docs(
            inputs=inputs,
            doc_sep_token=doc_sep_token,
            k=num_docs + 1,
            query=inputs[0],
        )
    # Choose a k greater than the total number of documents WITHOUT a query
    with pytest.raises(ValueError):
        _ = perturbations._randomly_sample_docs(
            inputs=inputs,
            doc_sep_token=doc_sep_token,
            k=(num_docs * 2) + 1,
        )

    # Randomly sample documents WITH a query
    random_docs = perturbations._randomly_sample_docs(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        k=num_docs - 1,
        query=inputs[0],
    )
    assert len(random_docs) == num_docs - 1
    assert all(example.strip() not in random_docs for example in inputs[0].split(doc_sep_token))
    assert all(doc in " ".join(inputs) for doc in random_docs)

    # Randomly sample documents WITHOUT a query
    random_docs = perturbations._randomly_sample_docs(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        k=num_docs + 1,
    )
    assert len(random_docs) == num_docs + 1
    assert any(example.strip() in random_docs for example in inputs[0].split(doc_sep_token))
    assert any(example.strip() in random_docs for example in inputs[1].split(doc_sep_token))

    # Randomly sample without providing a k
    random_docs = perturbations._randomly_sample_docs(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
    )
    assert len(random_docs) == num_docs * 2
    assert all(example.strip() in random_docs for example in inputs[0].split(doc_sep_token))
    assert all(example.strip() in random_docs for example in inputs[1].split(doc_sep_token))


def test_lexically_sample_docs() -> None:
    # Need to load an actual dataset here to reliably get rouge scores > 0.
    doc_sep_token = "|||||"
    dataset = load_dataset("multi_news", split="validation")
    inputs = dataset["document"][:4]
    targets = dataset["summary"][:4]
    # Will use the first input as query, so num of available docs for sampling is from the second
    num_docs = [len(util.split_docs(example, doc_sep_token)) for example in inputs]

    # Choose a k greater than the total number of documents WITH a query
    with pytest.raises(ValueError):
        _ = perturbations._lexically_sample_docs(
            inputs=inputs,
            doc_sep_token=doc_sep_token,
            k=sum(num_docs[1:]) + 1,
            query=inputs[0],
            strategy="similar",
        )
    # Choose a k greater than the total number of documents WITHOUT a query
    with pytest.raises(ValueError):
        _ = perturbations._lexically_sample_docs(
            inputs=inputs,
            doc_sep_token=doc_sep_token,
            k=sum(num_docs) + 1,
            strategy="similar",
        )

    # Provide neither a query nor a target
    with pytest.raises(ValueError):
        _ = perturbations._lexically_sample_docs(
            inputs=inputs,
            doc_sep_token=doc_sep_token,
            strategy="similar",
        )

    # Sample documents with a query
    sampled_docs = perturbations._lexically_sample_docs(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        k=num_docs[1] - 1,
        query=inputs[0],
        strategy="similar",
    )
    assert len(sampled_docs) == num_docs[1] - 1
    assert all(example.strip() not in sampled_docs for example in inputs[0].split(doc_sep_token))
    assert all(doc in " ".join(inputs) for doc in sampled_docs)

    # Sample documents with a target
    sampled_docs = perturbations._lexically_sample_docs(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        k=sum(num_docs) - 1,
        target=targets[0],
        strategy="similar",
    )
    assert len(sampled_docs) == sum(num_docs) - 1
    assert any(example.strip() in sampled_docs for example in inputs[0].split(doc_sep_token))
    assert any(example.strip() in sampled_docs for example in inputs[1].split(doc_sep_token))


def test_sorting() -> None:
    # We need a large number of documents to make it unlikely a random sort gives us same order
    num_docs = 64
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]

    perturbed = perturbations.sorting(inputs, doc_sep_token=doc_sep_token)

    # Because the perturbation is random we check other properties of the perturbed inputs.
    for input_example, perturbed_example in zip(inputs, perturbed):
        # That the pertubation was actually applied
        assert input_example != perturbed_example
        # That that perturbed example has only the documents from the input
        assert perturbed_example.count(doc_sep_token) == input_example.count(doc_sep_token)
        for input_doc, perturbed_doc in zip(
            input_example.split(doc_sep_token), perturbed_example.split(doc_sep_token)
        ):
            assert input_doc.strip() in perturbed_example
            assert perturbed_doc.strip() in input_example

    # A simple example to see if shuffling with a non-random strategy works
    inputs = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
    targets = ["this is a story about a cat"]

    expected = [f"this is a story about a cat {doc_sep_token} this is a story about a dog"]
    actual = perturbations.sorting(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        strategy="similar",
    )
    assert expected == actual

    expected = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
    actual = perturbations.sorting(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        strategy="dissimilar",
    )
    assert expected == actual


def test_addition() -> None:
    num_docs = 8
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.addition(inputs, doc_sep_token=doc_sep_token, perturbed_frac=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for perturbed_frac in [0.1, 0.5, 1.0]:
        expected_num_perturbed = num_docs + math.ceil(perturbed_frac * num_docs)
        perturbed = perturbations.addition(inputs, doc_sep_token=doc_sep_token, perturbed_frac=perturbed_frac)
        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            actual_num_perturbed = len(util.split_docs(perturbed_example, doc_sep_token))

            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That all input examples are in the perturbed example
            assert input_example in perturbed_example
            # That the total document count increased by the expected amount
            assert expected_num_perturbed == actual_num_perturbed

    # A simple example to see if addition with a non-random strategy works
    inputs = [
        f"this is a story about a dog {doc_sep_token} this is a story about a cat",
        f"this is another story about a cat {doc_sep_token} this looks purposfully dissimilar",
    ]
    targets = ["this is a story about a cat", "this is a story about a cat"]

    expected = [
        f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token} this is another story about a cat",
        f"this is another story about a cat {doc_sep_token} this looks purposfully dissimilar {doc_sep_token} this is a story about a cat",
    ]
    actual = perturbations.addition(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="similar",
    )
    assert expected == actual

    expected = [
        f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token} this looks purposfully dissimilar",
        f"this is another story about a cat {doc_sep_token} this looks purposfully dissimilar {doc_sep_token} this is a story about a dog",
    ]
    actual = perturbations.addition(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="dissimilar",
    )
    assert expected == actual


def test_deletion() -> None:
    num_docs = 8
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.deletion(inputs, doc_sep_token=doc_sep_token, perturbed_frac=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for perturbed_frac in [0.1, 0.5, 1.0]:
        expected_num_perturbed = num_docs - math.ceil(perturbed_frac * num_docs)
        perturbed = perturbations.deletion(inputs, doc_sep_token=doc_sep_token, perturbed_frac=perturbed_frac)

        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            perturbed_docs = util.split_docs(perturbed_example, doc_sep_token)
            actual_num_perturbed = len(perturbed_docs)

            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That all perturbed docs are in the input example
            assert all(doc in input_example for doc in perturbed_docs)
            # That the total document count decreased by the expected amount
            assert expected_num_perturbed == actual_num_perturbed

    # A simple example to see if deletion with a non-random strategy works
    inputs = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
    targets = ["this is a story about a cat"]

    expected = ["this is a story about a dog"]
    actual = perturbations.deletion(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="similar",
    )
    assert expected == actual

    expected = ["this is a story about a cat"]
    actual = perturbations.deletion(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="dissimilar",
    )
    assert expected == actual


def test_duplication() -> None:
    num_docs = 8
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.duplication(inputs, doc_sep_token=doc_sep_token, perturbed_frac=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for perturbed_frac in [0.1, 0.5, 1.0]:
        expected_num_perturbed = math.ceil(perturbed_frac * num_docs)
        perturbed = perturbations.duplication(inputs, doc_sep_token=doc_sep_token, perturbed_frac=perturbed_frac)

        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That that perturbed example has only the documents from the input
            assert expected_num_perturbed == len(util.split_docs(perturbed_example, doc_sep_token)) - num_docs
            for input_doc, perturbed_doc in zip(
                input_example.split(doc_sep_token), perturbed_example.split(doc_sep_token)
            ):
                assert input_doc.strip() in perturbed_example
                assert perturbed_doc.strip() in input_example

    # A simple example to see if duplication with a non-random strategy works
    inputs = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
    targets = ["this is a story about a cat"]

    expected = [
        f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token} this is a story about a cat"
    ]
    actual = perturbations.duplication(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="similar",
    )
    assert expected == actual

    expected = [
        f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token} this is a story about a dog"
    ]
    actual = perturbations.duplication(
        inputs=inputs,
        doc_sep_token=doc_sep_token,
        targets=targets,
        perturbed_frac=0.10,
        strategy="dissimilar",
    )
    assert expected == actual


def test_backtranslation() -> None:
    # Use a lesser number of documents because the translation is slow
    num_docs = 4
    doc_sep_token = "<doc-sep>"

    # Create random text so that back translation doesn't create identical text.
    url = "https://www.mit.edu/~ecprice/wordlist.10000"
    words = requests.get(url).text.splitlines()
    inputs = [
        f" {doc_sep_token} ".join(" ".join(random.sample(words, 16)) for _ in range(num_docs)),
        f" {doc_sep_token} ".join(" ".join(random.sample(words, 16)) for _ in range(num_docs, num_docs * 2)),
    ]

    # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.backtranslation(inputs, doc_sep_token=doc_sep_token, perturbed_frac=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for perturbed_frac in [0.1, 0.5, 1.0]:
        expected_num_perturbed = math.ceil(perturbed_frac * num_docs)
        perturbed = perturbations.backtranslation(inputs, doc_sep_token=doc_sep_token, perturbed_frac=perturbed_frac)

        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            input_docs = util.split_docs(input_example, doc_sep_token)
            perturbed_docs = util.split_docs(perturbed_example, doc_sep_token)
            actual_num_perturbed = len([doc for doc in perturbed_docs if doc.strip() not in input_docs])

            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That the total document count did not change
            assert len(input_docs) == len(perturbed_docs)
            # That the expected number of documents were perturbed
            assert expected_num_perturbed == actual_num_perturbed
