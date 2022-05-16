import copy
import math

from retrieval_exploration import perturbations
from retrieval_exploration.common import util
import pytest


def test_sample_random_docs() -> None:
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]

    with pytest.raises(ValueError):
        perturbations._sample_random_docs(
            # Choose a k greater than the total number of documents
            inputs,
            doc_sep_token=doc_sep_token,
            k=(num_docs * 2) + 1,
        )

    with pytest.raises(ValueError):
        perturbations._sample_random_docs(
            # Choose a k greater than the total number of documents
            inputs,
            doc_sep_token=doc_sep_token,
            k=4,
            exclude=[0, 1],
        )


def test_random_shuffling() -> None:
    # We need a large number of documents to make it unlikely a random shuffle gives us same order
    num_docs = 128
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]

    perturbed = perturbations.random_shuffle(inputs, doc_sep_token=doc_sep_token)

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


def test_random_addition() -> None:
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where per_perturbed is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.random_addition(inputs, doc_sep_token=doc_sep_token, per_perturbed=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for per_perturbed in [0.22, 0.5, 1.0]:
        expected_num_perturbed = num_docs + math.ceil(per_perturbed * num_docs)
        perturbed = perturbations.random_addition(
            inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
        )
        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            actual_num_perturbed = len(util.split_docs(perturbed_example, doc_sep_token))

            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That all input examples are in the perturbed example
            assert input_example in perturbed_example
            # That the total document count increased by the expected amount
            assert expected_num_perturbed == actual_num_perturbed


def test_random_deletion() -> None:
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where per_perturbed is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.random_deletion(inputs, doc_sep_token=doc_sep_token, per_perturbed=0.0)
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for per_perturbed in [0.22, 0.5, 1.0]:
        expected_num_perturbed = num_docs - math.ceil(per_perturbed * num_docs)
        perturbed = perturbations.random_deletion(
            inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
        )

        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            actual_num_perturbed = len(util.split_docs(perturbed_example, doc_sep_token))

            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That all perturbed examples are in the input example
            assert perturbed_example in perturbed_example
            # That the total document count decreased by the expected amount
            assert expected_num_perturbed == actual_num_perturbed


def test_random_duplication() -> None:
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where per_perturbed is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.random_duplication(
        inputs, doc_sep_token=doc_sep_token, per_perturbed=0.0
    )
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for per_perturbed in [0.22, 0.5, 1.0]:
        expected_num_perturbed = math.ceil(per_perturbed * num_docs)
        perturbed = perturbations.random_duplication(
            inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
        )

        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            # That the pertubation was actually applied
            assert input_example != perturbed_example
            # That that perturbed example has only the documents from the input
            assert (
                expected_num_perturbed
                == len(util.split_docs(perturbed_example, doc_sep_token)) - num_docs
            )
            for input_doc, perturbed_doc in zip(
                input_example.split(doc_sep_token), perturbed_example.split(doc_sep_token)
            ):
                assert input_doc.strip() in perturbed_example
                assert perturbed_doc.strip() in input_example


def test_random_replacement() -> None:
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where per_perturbed is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = perturbations.random_replacement(
        inputs, doc_sep_token=doc_sep_token, per_perturbed=0.0
    )
    assert expected == actual

    # Test the cases where a fraction of documents should be perturbed.
    for per_perturbed in [0.22, 0.5, 1.0]:
        expected_num_perturbed = math.ceil(per_perturbed * num_docs)
        perturbed = perturbations.random_replacement(
            inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
        )
        # Because the perturbation is random we check other properties of the perturbed inputs.
        for input_example, perturbed_example in zip(inputs, perturbed):
            actual_num_perturbed = num_docs - sum(
                [doc in input_example for doc in perturbed_example.split(doc_sep_token)]
            )
            assert expected_num_perturbed == actual_num_perturbed
