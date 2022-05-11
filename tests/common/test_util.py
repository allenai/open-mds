import copy
import random
from typing import Callable
import pytest
from retrieval_exploration.common import util


def test_preprocess_multi_news() -> None:
    doc_sep_token = "<doc-sep>"

    # Test a simple case with two documents, where one is longer than the other
    docs = [
        "Document numero uno.",
        # Including a document separator token at the end. Some examples in multi-news do this,
        # so we should make sure it doesn't trip up our logic.
        f"Document numero dos. {util._DOC_SEP_TOKENS['multi_news']}",
    ]
    text = f" {util._DOC_SEP_TOKENS['multi_news']} ".join(docs)
    summary = "This can be anything"

    expected_text, expected_summary = (
        "Document numero uno. <doc-sep> Document numero dos.",
        "This can be anything",
    )
    actual_text, actual_summary = util.preprocess_multi_news(
        text=text, summary=summary, doc_sep_token=doc_sep_token
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


def test_perturbed() -> None:
    num_docs = 4
    doc_sep_token = "<doc-sep>"
    inputs = [
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs)),
        f" {doc_sep_token} ".join(f"Document {i}" for i in range(num_docs, num_docs * 2)),
    ]
    # Test a simple case where per_perturbed is 0.0 and so this is a no-op
    expected = copy.deepcopy(inputs)
    actual = util.perturb(inputs, doc_sep_token=doc_sep_token, per_perturbed=0.0)
    assert expected == actual

    # Test the case where half the documents should be replaced.
    per_perturbed = 0.5
    expected = per_perturbed * num_docs
    perturbed = util.perturb(inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed)
    for input_, perturbed_ in zip(inputs, perturbed):
        actual = num_docs - sum([doc in input_ for doc in perturbed_.split(doc_sep_token)])
        assert expected == actual

    # Test the case where all documents should be replaced.
    per_perturbed = 1.0
    expected = per_perturbed * num_docs
    perturbed = util.perturb(inputs, doc_sep_token=doc_sep_token, per_perturbed=per_perturbed)
    for input_, perturbed_ in zip(inputs, perturbed):
        actual = num_docs - sum([doc in input_ for doc in perturbed_.split(doc_sep_token)])
        assert expected == actual
