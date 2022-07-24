import copy
import math
import random
import warnings

import pytest
import requests
from retrieval_exploration import perturbations
from retrieval_exploration.common import util


class TestPerturber:
    def test_invalid_perturbation(self) -> None:
        with pytest.raises(ValueError):
            _ = perturbations.Perturber("this-is-invalid", doc_sep_token="<doc-sep>")

    def test_mismatch_input_and_target_len(self) -> None:
        with pytest.raises(ValueError):
            perturber = perturbations.Perturber("addition", doc_sep_token="<doc-sep>")
            _ = perturber(["document 1 <doc-sep> document 2"], targets=["document 3"], perturbed_frac=0.1)

    def test_falsey_perturbed_frac(self) -> None:
        inputs = ["document 1 <doc-sep> document 2"]
        perturber = perturbations.Perturber("addition", doc_sep_token="<doc-sep>")

        with warnings.catch_warnings(record=True) as w:
            perturbed_inputs = perturber(inputs, perturbed_frac=None)
            assert inputs == perturbed_inputs
            assert str(w[0].message).endswith("Inputs will be returned unchanged.")

    def test_unused_documents(self) -> None:
        inputs = ["document 1 <doc-sep> document 2"]
        documents = ["document 3 <doc-sep> document 4"]
        perturber = perturbations.Perturber("sorting", doc_sep_token="<doc-sep>")

        with warnings.catch_warnings(record=True) as w:
            _ = perturber(inputs, perturbed_frac=0.1, documents=documents)
            assert str(w[0].message).endswith("They will be ignored.")

    def test_falsey_query_and_target(self) -> None:
        documents = ["document 1 <doc-sep> document 2", "document 3 <doc-sep> document 4"]

        # Fail to provide query and/or target when strategy is not "random"
        perturber = perturbations.Perturber("addition", doc_sep_token="<doc-sep>", strategy="similar")
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=None, target=None, k=2)

    def test_invalid_k(self) -> None:
        documents = ["document 1 <doc-sep> document 2", "document 3 <doc-sep> document 4"]
        perturber = perturbations.Perturber("addition", doc_sep_token="<doc-sep>", strategy="similar")

        # Choose a k greater than the total number of documents WITH a query
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=documents[0], target=None, k=3)

        # Choose a k greater than the total number of documents WITHOUT a query
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=None, target=documents[0], k=5)

    def test_unused_target(self) -> None:
        documents = ["document 1 <doc-sep> document 2"]
        target = "Target text"
        perturber = perturbations.Perturber("addition", doc_sep_token="<doc-sep>", strategy="random")

        with warnings.catch_warnings(record=True) as w:
            _ = perturber._select_docs(documents, target=target, k=2)
            assert str(w[0].message).endswith("target will be ignored.")

    def test_select_docs(self) -> None:
        num_docs_per_example = 8
        doc_sep_token = "<doc-sep>"
        documents = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs_per_example)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs_per_example, num_docs_per_example * 2)),
        ]
        target = "Target text"
        num_docs = [util.get_num_docs(example, doc_sep_token) for example in documents]

        for strategy in ["random", "best-case", "worst-case"]:

            perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token, strategy=strategy)

            if strategy == "random":
                sampled_docs = perturber._select_docs(documents, query=None, target=None, k=sum(num_docs) - 1)
                assert len(sampled_docs) == sum(num_docs) - 1
                assert all(doc.strip() in " ".join(documents) for doc in sampled_docs)

            # Sample documents with a query
            sampled_docs = perturber._select_docs(
                documents,
                k=num_docs[1] - 1,
                query=documents[0],
            )
            assert len(sampled_docs) == num_docs[1] - 1
            assert all(example.strip() not in sampled_docs for example in documents[0].split(doc_sep_token))
            assert all(doc in " ".join(documents) for doc in sampled_docs)

            # Sample documents with a target
            sampled_docs = perturber._select_docs(
                documents,
                k=sum(num_docs) - 1,
                target=target,
            )
            assert len(sampled_docs) == sum(num_docs) - 1
            assert all(doc.strip() in " ".join(documents) for doc in sampled_docs)

            # Sample documents with a both a query and a target
            sampled_docs = perturber._select_docs(
                documents,
                k=num_docs[1] - 1,
                query=documents[0],
                target=target,
            )
            assert len(sampled_docs) == num_docs[1] - 1
            assert all(example.strip() not in sampled_docs for example in documents[0].split(doc_sep_token))
            assert all(doc in " ".join(documents) for doc in sampled_docs)

    def test_backtranslation(self) -> None:
        # Use a lesser number of documents because the translation is slow
        num_docs = 4
        doc_sep_token = "<doc-sep>"

        # Create random text so that back translation doesn't create identical text.
        url = "https://www.mit.edu/~ecprice/wordlist.10000"
        words = requests.get(url).text.splitlines()
        inputs = [
            f" {doc_sep_token} ".join(" ".join(random.sample(words, 24)) for _ in range(num_docs)),
            f" {doc_sep_token} ".join(" ".join(random.sample(words, 24)) for _ in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("backtranslation", doc_sep_token=doc_sep_token)

        # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
        expected = copy.deepcopy(inputs)
        actual = perturber(inputs, perturbed_frac=0.0)
        assert expected == actual

        # Test the cases where a fraction of documents should be perturbed.
        for perturbed_frac in [0.1, 0.5, 1.0]:
            expected_num_perturbed = math.ceil(perturbed_frac * num_docs)
            perturbed = perturber(inputs, perturbed_frac=perturbed_frac)

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

        # A simple example to see if backtranslation with a non-random strategy works. Because it is difficult to know
        # in advance what the backtranslation will look like, we check whether the expected document is different than
        # the original document. Need to choose complicated sentences to guarantee backtranslation is not perfect.
        inputs = [f"Throw a spanner in the works {doc_sep_token} A different kettle of fish"]
        targets = ["Like shooting fish in a barrel"]

        input_docs = util.split_docs(inputs[0], doc_sep_token=doc_sep_token)

        # best-case
        perturber = perturbations.Perturber("backtranslation", doc_sep_token=doc_sep_token, strategy="best-case")
        actual = perturber(
            inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        actual_docs = util.split_docs(actual[0], doc_sep_token=doc_sep_token)
        assert actual_docs[0] != input_docs[0]
        assert actual_docs[1] == input_docs[1]

        # worst-case
        perturber = perturbations.Perturber("backtranslation", doc_sep_token=doc_sep_token, strategy="worst-case")
        actual = perturber(
            inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        actual_docs = util.split_docs(actual[0], doc_sep_token=doc_sep_token)
        assert actual_docs[0] == input_docs[0]
        assert actual_docs[1] != input_docs[1]

    def test_sorting(self) -> None:
        # We need a large number of documents to make it unlikely a random sort gives us same order
        num_docs = 64
        doc_sep_token = "<doc-sep>"
        inputs = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("sorting", doc_sep_token=doc_sep_token)
        perturbed = perturber(inputs)

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

        # best-case
        perturber = perturbations.Perturber("sorting", doc_sep_token=doc_sep_token, strategy="best-case")
        expected = [f"this is a story about a cat {doc_sep_token} this is a story about a dog"]
        actual = perturber(inputs, targets=targets)
        assert expected == actual

        # worst-case
        perturber = perturbations.Perturber("sorting", doc_sep_token=doc_sep_token, strategy="worst-case")
        expected = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
        actual = perturber(inputs, targets=targets)
        assert expected == actual

    def test_duplication(self) -> None:
        num_docs = 8
        doc_sep_token = "<doc-sep>"
        inputs = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("duplication", doc_sep_token=doc_sep_token)

        # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
        expected = copy.deepcopy(inputs)
        actual = perturber(inputs, perturbed_frac=0.0)
        assert expected == actual

        # Test the cases where a fraction of documents should be perturbed.
        for perturbed_frac in [0.1, 0.5, 1.0]:
            expected_num_perturbed = math.ceil(perturbed_frac * num_docs)
            perturbed = perturber(inputs, perturbed_frac=perturbed_frac)

            # Because the perturbation is random we check other properties of the perturbed inputs.
            for input_example, perturbed_example in zip(inputs, perturbed):
                # That the pertubation was actually applied
                assert input_example != perturbed_example
                # That that perturbed example has only the documents from the input
                assert expected_num_perturbed == util.get_num_docs(perturbed_example, doc_sep_token) - num_docs
                for input_doc, perturbed_doc in zip(
                    input_example.split(doc_sep_token), perturbed_example.split(doc_sep_token)
                ):
                    assert input_doc.strip() in perturbed_example
                    assert perturbed_doc.strip() in input_example

        # A simple example to see if duplication with a non-random strategy works
        inputs = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
        targets = ["this is a story about a cat"]

        # best-case
        perturber = perturbations.Perturber("duplication", doc_sep_token=doc_sep_token, strategy="best-case")
        expected = [
            (
                f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token}"
                " this is a story about a cat"
            )
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

        # worst-case
        perturber = perturbations.Perturber("duplication", doc_sep_token=doc_sep_token, strategy="worst-case")
        expected = [
            (
                f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token}"
                " this is a story about a dog"
            )
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

    def test_addition(self) -> None:
        num_docs = 8
        doc_sep_token = "<doc-sep>"
        inputs = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token)

        # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
        expected = copy.deepcopy(inputs)
        actual = perturber(inputs, perturbed_frac=0.0)
        assert expected == actual

        # Test the cases where a fraction of documents should be perturbed.
        for perturbed_frac in [0.1, 0.5, 1.0]:
            expected_num_perturbed = num_docs + math.ceil(perturbed_frac * num_docs)
            perturbed = perturber(inputs, perturbed_frac=perturbed_frac)
            # Because the perturbation is random we check other properties of the perturbed inputs.
            for input_example, perturbed_example in zip(inputs, perturbed):
                actual_num_perturbed = util.get_num_docs(perturbed_example, doc_sep_token)

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

        # best-case
        perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token, strategy="best-case")
        expected = [
            (
                f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token}"
                " this is another story about a cat"
            ),
            (
                f"this is another story about a cat {doc_sep_token} this looks purposfully dissimilar {doc_sep_token}"
                " this is a story about a cat"
            ),
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

        # worst-case
        perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token, strategy="worst-case")
        expected = [
            (
                f"this is a story about a dog {doc_sep_token} this is a story about a cat {doc_sep_token}"
                " this looks purposfully dissimilar"
            ),
            (
                f"this is another story about a cat {doc_sep_token} this looks purposfully dissimilar {doc_sep_token}"
                " this is a story about a dog"
            ),
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

    def test_deletion(self) -> None:
        num_docs = 8
        doc_sep_token = "<doc-sep>"
        inputs = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("deletion", doc_sep_token=doc_sep_token)

        # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
        expected = copy.deepcopy(inputs)
        actual = perturber(inputs, perturbed_frac=0.0)
        assert expected == actual

        # Test the cases where a fraction of documents should be perturbed.
        for perturbed_frac in [0.1, 0.5, 1.0]:
            expected_num_remaining = num_docs - math.ceil(perturbed_frac * num_docs)
            perturbed = perturber(inputs, perturbed_frac=perturbed_frac)

            # Because the perturbation is random we check other properties of the perturbed inputs.
            for input_example, perturbed_example in zip(inputs, perturbed):
                perturbed_docs = util.split_docs(perturbed_example, doc_sep_token=doc_sep_token)
                actual_num_remaining = util.get_num_docs(perturbed_example, doc_sep_token=doc_sep_token)

                # That the pertubation was actually applied
                assert input_example != perturbed_example
                # That all perturbed docs are in the input example
                assert all(doc in input_example for doc in perturbed_docs if doc)
                # That the total document count decreased by the expected amount
                assert expected_num_remaining == actual_num_remaining

        # A simple example to see if deletion with a non-random strategy works
        inputs = [f"this is a story about a dog {doc_sep_token} this is a story about a cat"]
        targets = ["this is a story about a cat"]

        # best-case
        perturber = perturbations.Perturber("deletion", doc_sep_token=doc_sep_token, strategy="best-case")
        expected = ["this is a story about a cat"]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

        # worst-case
        perturber = perturbations.Perturber("deletion", doc_sep_token=doc_sep_token, strategy="worst-case")
        expected = ["this is a story about a dog"]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

    def test_replacement(self) -> None:
        num_docs = 8
        doc_sep_token = "<doc-sep>"
        inputs = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
        ]

        perturber = perturbations.Perturber("replacement", doc_sep_token=doc_sep_token)

        # Test a simple case where perturbed_frac is 0.0 and so this is a no-op
        expected = copy.deepcopy(inputs)
        actual = perturber(inputs, perturbed_frac=0.0)
        assert expected == actual

        # Test the cases where a fraction of documents should be perturbed.
        for perturbed_frac in [0.1, 0.5, 1.0]:
            expected_num_perturbed = math.ceil(perturbed_frac * num_docs)
            perturbed = perturber(inputs, perturbed_frac=perturbed_frac)

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

        # A simple example to see if replacement with a non-random strategy works
        inputs = [
            f"this is a story about a dog {doc_sep_token} this is a story about a cat",
            "this is a story about a siamese cat",
            "this is a story about something else",
        ]
        targets = ["this is a story about a cat", "this is a story about a dog", "this is a story about a dog"]

        # best-case
        perturber = perturbations.Perturber("replacement", doc_sep_token=doc_sep_token, strategy="best-case")
        expected = [
            f"this is a story about a siamese cat {doc_sep_token} this is a story about a cat",
            "this is a story about a dog",
            "this is a story about a dog",
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual

        # worst-case
        perturber = perturbations.Perturber("replacement", doc_sep_token=doc_sep_token, strategy="worst-case")
        expected = [
            f"this is a story about a dog {doc_sep_token} this is a story about something else",
            "this is a story about something else",
            "this is a story about a siamese cat",
        ]
        actual = perturber(
            inputs=inputs,
            perturbed_frac=0.10,
            targets=targets,
        )
        assert expected == actual
