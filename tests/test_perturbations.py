import copy
import math
import random
import warnings

import pytest
import requests
from retrieval_exploration import perturbations
from retrieval_exploration.common import util


class TestPerturber:
    @pytest.mark.parametrize("strategy", ["random", "best-case", "worst-case"])
    def test_invalid_perturbation(self, strategy: str) -> None:
        with pytest.raises(ValueError):
            _ = perturbations.Perturber("this-is-invalid", doc_sep_token="<doc-sep>", strategy=strategy)

    @pytest.mark.parametrize(
        "perturbation", ["backtranslation", "sorting", "duplication", "addition", "deletion", "replacement"]
    )
    def test_invalid_strategy(self, perturbation: str) -> None:
        with pytest.raises(ValueError):
            perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>", strategy="this-is-invalid")

    @pytest.mark.parametrize(
        "perturbation", ["backtranslation", "sorting", "duplication", "addition", "deletion", "replacement"]
    )
    def test_mismatch_input_and_target_len(self, perturbation: str) -> None:
        with pytest.raises(ValueError):
            perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>")
            _ = perturber(
                ["document 1 <doc-sep> document 2"], targets=["document 3", "document 4"], perturbed_frac=0.1
            )

    @pytest.mark.parametrize("perturbation", ["backtranslation", "duplication", "addition", "deletion", "replacement"])
    def test_falsey_perturbed_frac(self, perturbation: str) -> None:
        inputs = ["document 1 <doc-sep> document 2"]
        perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>")

        with warnings.catch_warnings(record=True) as w:
            perturbed_inputs = perturber(inputs, perturbed_frac=None)
            assert inputs == perturbed_inputs
            assert str(w[0].message).endswith("Inputs will be returned unchanged.")

    @pytest.mark.parametrize("perturbation", ["backtranslation", "duplication", "deletion"])
    def test_unused_documents(self, perturbation: str) -> None:
        inputs = ["document 1 <doc-sep> document 2"]
        documents = ["document 3 <doc-sep> document 4"]
        perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>")

        with warnings.catch_warnings(record=True) as w:
            _ = perturber(inputs, perturbed_frac=0.1, documents=documents)
            assert str(w[0].message).endswith("They will be ignored.")

    @pytest.mark.parametrize(
        "perturbation", ["backtranslation", "sorting", "duplication", "addition", "deletion", "replacement"]
    )
    @pytest.mark.parametrize("strategy", ["best-case", "worst-case"])
    def test_falsey_query_and_target(self, perturbation: str, strategy: str) -> None:
        documents = ["document 1 <doc-sep> document 2", "document 3 <doc-sep> document 4"]

        # Fail to provide query and/or target when strategy is not "random"
        perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>", strategy=strategy)
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=None, target=None, k=2)

    @pytest.mark.parametrize("strategy", ["best-case", "worst-case"])
    @pytest.mark.parametrize("perturbation", ["backtranslation", "duplication", "addition", "deletion", "replacement"])
    def test_invalid_k(self, strategy: str, perturbation: str) -> None:
        documents = ["document 1 <doc-sep> document 2", "document 3 <doc-sep> document 4"]
        perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>", strategy=strategy)

        # Choose a k greater than the total number of documents WITH a query
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=documents[0], target=None, k=3)

        # Choose a k greater than the total number of documents WITHOUT a query
        with pytest.raises(ValueError):
            _ = perturber._select_docs(documents, query=None, target=documents[0], k=5)

    @pytest.mark.parametrize(
        "perturbation", ["backtranslation", "sorting", "duplication", "addition", "deletion", "replacement"]
    )
    def test_unused_target(self, perturbation: str) -> None:
        documents = ["document 1 <doc-sep> document 2"]
        target = "Target text"
        perturber = perturbations.Perturber(perturbation, doc_sep_token="<doc-sep>", strategy="random")

        with warnings.catch_warnings(record=True) as w:
            _ = perturber._select_docs(documents, target=target, k=2)
            assert str(w[0].message).endswith("target will be ignored.")

    def test_select_docs_random(self) -> None:
        num_docs_per_example = 8
        doc_sep_token = "<doc-sep>"
        documents = [
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs_per_example)),
            f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs_per_example, num_docs_per_example * 2)),
        ]
        target = "Target text"
        num_docs = [util.get_num_docs(example, doc_sep_token) for example in documents]

        # select_docs function is independent of perturbation type, so we can use any valid perturbation here
        perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token, strategy="random")

        # Select documents given no query and no target
        sampled_docs = perturber._select_docs(documents, query=None, target=None, k=2)
        assert len(sampled_docs) == 2
        assert all(doc.strip() in " ".join(documents) for doc in sampled_docs)

        # Select documents given a query
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
            k=2,
            target=target,
        )
        assert len(sampled_docs) == 2
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

    @pytest.mark.parametrize("strategy", ["best-case", "worst-case"])
    def test_select_docs(self, strategy: str) -> None:
        doc_sep_token = "<doc-sep>"
        documents = [
            (
                f"A mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms {doc_sep_token}"
                f" Mitochondria use aerobic respiration to generate most of the cell's supply of adenosine triphosphate (ATP) {doc_sep_token}"
                f" Mitochondria are commonly between 0.75 and 3 μm2 in area, but vary considerably in size and structure. {doc_sep_token}"
                f" Stefani Joanne Angelina Germanotta, known professionally as Lady Gaga, is an American singer, songwriter, and actress. {doc_sep_token}"
                f" Gaga's five succeeding studio albums all debuted atop the US Billboard 200. {doc_sep_token}"
            )
        ]

        # select_docs function is independent of perturbation type, or strategy
        perturber = perturbations.Perturber("addition", doc_sep_token=doc_sep_token, strategy=strategy)

        # Select documents given a query (largest == True)
        query, *remaining = util.split_docs(documents[0], doc_sep_token)
        expected = remaining[:2]
        sampled_docs = perturber._select_docs(
            documents,
            k=2,
            query=query,
            largest=True,
        )
        assert len(sampled_docs) == 2
        # Check that the query was excluded
        assert query.strip() not in [doc.strip() for doc in sampled_docs]
        # Check that the sampled documents are the most similar remaining documents
        assert all(doc.strip() in expected for doc in sampled_docs)

        # Select documents given a query (largest == False)
        query, *remaining = util.split_docs(documents[0], doc_sep_token)
        expected = remaining[-2:]
        sampled_docs = perturber._select_docs(documents, k=2, query=query, largest=False)
        assert len(sampled_docs) == 2
        # Check that the query was excluded
        assert query.strip() not in [doc.strip() for doc in sampled_docs]
        # Check that the sampled documents are the most similar remaining documents
        assert all(doc.strip() in expected for doc in sampled_docs)

        # Sample documents with a target (largest == True)
        target = "The number of mitochondria in a cell can vary widely by organism, tissue, and cell type."
        expected = util.split_docs(documents[0], doc_sep_token)[:3]
        sampled_docs = perturber._select_docs(
            documents,
            k=3,
            target=target,
            largest=True,
        )
        assert len(sampled_docs) == 3
        assert all(doc.strip() in expected for doc in sampled_docs)

        # Sample documents with a target (largest == False)
        target = "The number of mitochondria in a cell can vary widely by organism, tissue, and cell type."
        expected = util.split_docs(documents[0], doc_sep_token)[-2:]
        sampled_docs = perturber._select_docs(
            documents,
            k=2,
            target=target,
            largest=False,
        )
        assert len(sampled_docs) == 2
        assert all(doc.strip() in expected for doc in sampled_docs)

        # Sample documents with a both a query and a target (largest == True)
        target = "Gaga began performing as a teenager, singing at open mic nights and acting in school plays."
        query, *remaining = util.split_docs(documents[0], doc_sep_token)
        expected = remaining[-2:]
        sampled_docs = perturber._select_docs(
            documents,
            k=2,
            query=query,
            target=target,
            largest=True,
        )
        assert len(sampled_docs) == 2
        assert all(doc.strip() in expected for doc in sampled_docs)

        # Sample documents with a both a query and a target (largest == False)
        target = "Gaga began performing as a teenager, singing at open mic nights and acting in school plays."
        query, *remaining = util.split_docs(documents[0], doc_sep_token)
        expected = remaining[:2]
        sampled_docs = perturber._select_docs(
            documents,
            k=2,
            query=query,
            target=target,
            largest=False,
        )
        assert len(sampled_docs) == 2
        assert all(doc.strip() in expected for doc in sampled_docs)

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
