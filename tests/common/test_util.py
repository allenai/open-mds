from typing import Callable, List

import pytest
from retrieval_exploration.common import util


def test_split_docs() -> None:
    doc_sep_token = "<doc-sep>"

    # Test the case with an empty string as input
    expected: List[str] = []
    actual = util.split_docs("", doc_sep_token=doc_sep_token)
    assert expected == actual

    actual = util.split_docs("This does not contain doc_sep_token", doc_sep_token=doc_sep_token)
    expected = ["This does not contain doc_sep_token"]
    assert expected == actual


def test_preprocess_multi_news() -> None:
    doc_sep_token = "<doc-sep>"

    docs = [
        "Document numero uno.",
        # Including a document separator token at the end. Some examples in multi-news do this,
        # so we should make sure it doesn't trip up our logic.
        f"Document numero dos. {util._DOC_SEP_TOKENS['multi_news']}",
    ]

    document = f" {util._DOC_SEP_TOKENS['multi_news']} ".join(docs)
    summary = "This is the summary."

    expected_text, expected_summary = (
        "Document numero uno. <doc-sep> Document numero dos.",
        "This is the summary.",
    )
    actual_text, actual_summary = util.preprocess_multi_news(
        text=document, summary=summary, doc_sep_token=doc_sep_token
    )
    assert expected_text == actual_text
    assert expected_summary == actual_summary


def test_preprocess_multi_x_science_sum() -> None:
    doc_sep_token = "<doc-sep>"

    abstract = "This is the query abstract."
    ref_abstracts = {"abstract": ["This is a cited abstract."]}
    related_work = "This is the related work."

    expected_text, expected_summary = (
        "This is the query abstract. <doc-sep> This is a cited abstract.",
        "This is the related work.",
    )
    actual_text, actual_summary = util.preprocess_multi_x_science_sum(
        text=abstract,
        summary=related_work,
        ref_abstracts=ref_abstracts,
        doc_sep_token=doc_sep_token,
    )
    assert expected_text == actual_text
    assert expected_summary == actual_summary


def test_get_doc_sep_token(hf_tokenizer: Callable) -> None:
    # A model from the PRIMERA family
    tokenizer = hf_tokenizer("allenai/PRIMERA")
    assert util.get_doc_sep_token(tokenizer) == util._DOC_SEP_TOKENS["primera"]

    # A model which defines a sep_token
    tokenizer = hf_tokenizer("bert-base-cased")
    assert util.get_doc_sep_token(tokenizer) == tokenizer.sep_token

    # A model which only defines an eos_token
    tokenizer = hf_tokenizer("t5-small")
    assert util.get_doc_sep_token(tokenizer) == tokenizer.eos_token

    # Check that a ValueError is raised if no suitable token is found
    tokenizer = hf_tokenizer("t5-small")
    tokenizer.eos_token = None
    with pytest.raises(ValueError):
        _ = util.get_doc_sep_token(tokenizer)


def test_truncate_multi_doc(hf_tokenizer: Callable) -> None:
    max_length = 24
    doc_sep_token = "<doc-sep>"
    tokenizer = hf_tokenizer("allenai/PRIMERA")

    # Test a simple case with two documents, where one is longer than the other
    docs = [
        "I am document one.",
        # Including a document separator token at the end. Some examples in multi-news do this,
        # so we should make sure it doesn't trip up logic.
        f"I am document two. I am a little longer than document one. {doc_sep_token}",
    ]
    text = f" {doc_sep_token} ".join(docs)

    expected = "I am document one. <doc-sep> I am document two. I am a little longer"
    actual = util.truncate_multi_doc(
        text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer
    )
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length

    # Test a simple case with two documents, where both are the same length
    docs = [
        "I am document one. I am the same length as document two",
        "I am document two. I am the same length as document one.",
    ]
    text = f" {doc_sep_token} ".join(docs)

    expected = (
        "I am document one. I am the same length <doc-sep> I am document two. I am the same length"
    )
    actual = util.truncate_multi_doc(
        text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer
    )
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length

    # Test that the argument num_docs is respected
    docs = [
        "I am document one. I am the same length as document two",
        "I am document two. I am the same length as document one.",
        "I am document three. I am the same length as both document one and two.",
    ]
    text = f" {doc_sep_token} ".join(docs)

    expected = (
        "I am document one. I am the same length <doc-sep>"
        " I am document two. I am the same length <doc-sep>"
        " I am document three. I am the same length"
    )
    actual = util.truncate_multi_doc(
        text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer, num_docs=2
    )
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length

    expected = (
        "I am document one. I <doc-sep> I am document two. I <doc-sep> I am document three. I"
    )
    actual = util.truncate_multi_doc(
        text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer, num_docs=3
    )
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length


def test_get_global_attention_mask() -> None:
    # Test a simple case with two tokens to be globally attended to
    input_ids = [[117, 0, 6, 42], [0, 2, 117, 24]]
    token_ids = [117, 42]
    expected_global_attention_mask = [[1, 0, 0, 1], [0, 0, 1, 0]]
    actual_global_attention_mask = util.get_global_attention_mask(
        input_ids=input_ids, token_ids=token_ids
    )
    assert expected_global_attention_mask == actual_global_attention_mask

    # Test the case when input_ids is empty
    actual_global_attention_mask = util.get_global_attention_mask(input_ids=[], token_ids=token_ids)
    expected_global_attention_mask = []
    assert expected_global_attention_mask == actual_global_attention_mask

    # Test the case when token_ids is empty
    actual_global_attention_mask = util.get_global_attention_mask(input_ids=input_ids, token_ids=[])
    expected_global_attention_mask = [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert expected_global_attention_mask == actual_global_attention_mask


def test_get_num_original_docs():
    num_docs = 16
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
    ]

    # Test the case where no perturbations are applied
    assert util.get_num_original_docs(inputs, doc_sep_token) == [num_docs]
    assert util.get_num_original_docs(
        inputs, doc_sep_token, perturbation="addition", per_perturbed=None
    ) == [num_docs]

    # Test the case with addition
    expected = [num_docs - 2]
    actual = util.get_num_original_docs(
        inputs, doc_sep_token, perturbation="addition", per_perturbed=0.10
    )
    assert expected == actual

    # Test the case with deletion
    expected = [num_docs + 2]
    actual = util.get_num_original_docs(
        inputs, doc_sep_token, perturbation="deletion", per_perturbed=0.1
    )
    assert expected == actual

    # Test the case where the inputs are a string
    expected = [num_docs + 2]
    actual = util.get_num_original_docs(
        inputs[0], doc_sep_token, perturbation="deletion", per_perturbed=0.1
    )
    assert expected == actual
