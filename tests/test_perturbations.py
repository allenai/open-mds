import copy
from retrieval_exploration import perturbations


def test_random_replacement() -> None:
    num_docs = 4
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

    # Test the case where per_perturbed results in a whole number of documents to replace.
    per_perturbed = 0.5
    expected_num_perturbed = per_perturbed * num_docs
    perturbed = perturbations.random_replacement(
        inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
    )
    for input_, perturbed_ in zip(inputs, perturbed):
        actual_num_perturbed = num_docs - sum(
            [doc in input_ for doc in perturbed_.split(doc_sep_token)]
        )
        assert expected_num_perturbed == actual_num_perturbed

    # Test the case where all documents should be replaced.
    per_perturbed = 1.0
    expected_num_perturbed = per_perturbed * num_docs
    perturbed = perturbations.random_replacement(
        inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed
    )
    for input_, perturbed_ in zip(inputs, perturbed):
        actual_num_perturbed = num_docs - sum(
            [doc in input_ for doc in perturbed_.split(doc_sep_token)]
        )
        assert expected_num_perturbed == actual_num_perturbed


def test_random_shuffling() -> None:
    # We need a large number of documents to make it unlikely random shuffle gives us the same order
    num_docs = 128
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    perturbed = perturbations.random_shuffle(inputs, doc_sep_token=doc_sep_token)

    # Because the shuffling is random we check other properties of the perturbed inputs.
    for input_example, perturbed_example in zip(inputs, perturbed):
        # That the pertubation was actually applied
        assert input_example != perturbed_example
        # That that perturbed example has only the documents from the input
        assert perturbed_example.count(doc_sep_token) == input_example.count(doc_sep_token)
        for document in input_example.split(doc_sep_token):
            assert document.strip() in perturbed_example
