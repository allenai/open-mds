import re
import sys
import warnings
from typing import Callable, List

import pytest
from hypothesis import given
from hypothesis.strategies import booleans, text
from retrieval_exploration.common import util
from transformers import AutoTokenizer


@given(text=text(), lowercase=booleans())
def test_sanitize_text(text: str, lowercase: bool) -> None:
    sanitized = util.sanitize_text(text, lowercase=lowercase)

    # There should be no cases of multiple spaces or tabs
    assert re.search(r"[ ]{2,}", sanitized) is None
    assert "\t" not in sanitized
    # The beginning and end of the string should be stripped of whitespace
    assert not sanitized.startswith(("\n", " "))
    assert not sanitized.endswith(("\n", " "))
    # Sometimes, hypothesis generates text that cannot be lowercased (like latin characters).
    # We don't particularly care about this, and it breaks this check.
    # Only run if the generated text can be lowercased.
    if lowercase and text.lower().islower():
        assert all(not char.isupper() for char in sanitized)


def test_parse_omega_conf() -> None:
    # Simulate command line arguments
    sys.argv = [
        "this should be ignored",
        "test_fixtures/conf/base.yml",
        "test_fixtures/conf/extension.yml",
        "argument_4=changed by cli!",
    ]
    expected = {
        "argument_1": "argument 1",
        "argument_2": "changed by extension.yml!",
        "argument_3": "argument 3",
        "argument_4": "changed by cli!",
    }
    actual = util.parse_omega_conf()
    assert expected == actual


def test_jaccard_similarity_score() -> None:
    # Both strings cannot be empty
    with warnings.catch_warnings(record=True) as w:
        assert util.jaccard_similarity_score("", "") == 1.0
        assert len(w) == 1

    # One string is empty
    assert util.jaccard_similarity_score("", "hello") == 0.0

    # Strings are identical
    assert util.jaccard_similarity_score("hello", "hello") == 1.0

    # String are non-identical
    assert util.jaccard_similarity_score("hello world", "hello you") == 1 / 3

    # Check that punctation is treated as its own token
    assert util.jaccard_similarity_score("hello world", "hello world!") == 2 / 3


def test_split_docs() -> None:
    doc_sep_token = "<doc-sep>"

    # Test the case with an empty string as input
    expected: List[str] = [""]
    actual = util.split_docs("", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = ["Document 1", "Document 2"]
    # Include whitespace to ensure that is handled correctly
    actual = util.split_docs(f"  Document 1 {doc_sep_token} Document 2  ", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = ["This does not contain doc_sep_token"]
    actual = util.split_docs("  This does not contain doc_sep_token  ", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = ["This is ends with characters from doc_sep_token sep"]
    actual = util.split_docs("  This is ends with characters from doc_sep_token sep", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = ["This is ends with the doc-sep token"]
    actual = util.split_docs("  This is ends with the doc-sep token <doc-sep>  ", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = ["Document 1", "This is ends with multiple doc-sep tokens"]
    actual = util.split_docs(
        "  Document 1 <doc-sep> This is ends with multiple doc-sep tokens <doc-sep>  <doc-sep>",
        doc_sep_token=doc_sep_token,
    )
    assert expected == actual


def test_get_num_docs() -> None:
    doc_sep_token = "<doc-sep>"

    # Test the case with an empty string as input
    expected = 0
    actual = util.get_num_docs("", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = 2
    actual = util.get_num_docs(f"Document 1 {doc_sep_token} Document 2", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = 1
    actual = util.get_num_docs("This does not contain doc_sep_token", doc_sep_token=doc_sep_token)
    assert expected == actual

    expected = 1
    actual = util.get_num_docs("This is ends with characters from doc_sep_token sep", doc_sep_token=doc_sep_token)
    assert expected == actual


def test_get_doc_sep_token(hf_tokenizer: Callable) -> None:
    # A model from the PRIMERA family
    tokenizer = hf_tokenizer("allenai/PRIMERA")
    assert util.get_doc_sep_token(tokenizer) == util.DOC_SEP_TOKENS["primera"]

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


def test_get_global_attention_mask() -> None:
    # Test a simple case with two tokens to be globally attended to
    input_ids = [[117, 0, 6, 42], [0, 2, 117, 24]]
    token_ids = [117, 42]
    expected_global_attention_mask = [[1, 0, 0, 1], [0, 0, 1, 0]]
    actual_global_attention_mask = util.get_global_attention_mask(input_ids=input_ids, token_ids=token_ids)
    assert expected_global_attention_mask == actual_global_attention_mask

    # Test the case when input_ids is empty
    actual_global_attention_mask = util.get_global_attention_mask(input_ids=[], token_ids=token_ids)
    expected_global_attention_mask = []
    assert expected_global_attention_mask == actual_global_attention_mask

    # Test the case when token_ids is empty
    actual_global_attention_mask = util.get_global_attention_mask(input_ids=input_ids, token_ids=[])
    expected_global_attention_mask = [[0, 0, 0, 0], [0, 0, 0, 0]]
    assert expected_global_attention_mask == actual_global_attention_mask


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
    actual = util.truncate_multi_doc(text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer)
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length

    # Test a simple case with two documents, where both are the same length
    docs = [
        "I am document one. I am the same length as document two",
        "I am document two. I am the same length as document one.",
    ]
    text = f" {doc_sep_token} ".join(docs)

    expected = "I am document one. I am the same length <doc-sep> I am document two. I am the same length"
    actual = util.truncate_multi_doc(text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer)
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

    expected = "I am document one. I <doc-sep> I am document two. I <doc-sep> I am document three. I"
    actual = util.truncate_multi_doc(
        text, doc_sep_token=doc_sep_token, max_length=max_length, tokenizer=tokenizer, num_docs=3
    )
    assert expected == actual
    assert len(tokenizer(text, max_length=max_length)["input_ids"]) == max_length


def test_batch_decode_multi_doc() -> None:
    # A tokenizer with both bos and eos tokens
    tokenizer = AutoTokenizer.from_pretrained("allenai/PRIMERA")
    doc_sep_token = tokenizer.additional_special_tokens[0]
    inputs = [f"This is a test {doc_sep_token} with a doc-sep token {tokenizer.pad_token}"]
    input_ids = tokenizer(inputs).input_ids

    # Function should complain if skip_special_tokens is True
    with warnings.catch_warnings(record=True) as w:
        _ = util.batch_decode_multi_doc(input_ids, tokenizer, doc_sep_token=doc_sep_token, skip_special_tokens=True)
        assert (
            str(w[0].message)
            == "`skip_special_tokens=True` was provided to batch_decode_multi_doc but will be ignored."
        )

    # When doc_sep_token != bos_token or eos_token
    expected = [f"This is a test {doc_sep_token} with a doc-sep token"]
    actual = util.batch_decode_multi_doc(input_ids, tokenizer, doc_sep_token=doc_sep_token)
    assert expected == actual

    # When doc_sep_token == bos_token or eos_token
    expected = [f"{tokenizer.bos_token}This is a test {doc_sep_token} with a doc-sep token"]
    actual = util.batch_decode_multi_doc(input_ids, tokenizer, doc_sep_token=tokenizer.bos_token)
    assert expected == actual


def test_preprocess_multi_news() -> None:
    doc_sep_token = "<doc-sep>"

    docs = [
        "Document numero uno.",
        # Including a document separator token at the end. Some examples in multi-news do this,
        # so we should make sure it doesn't trip up our logic.
        f"Document numero dos. {util.DOC_SEP_TOKENS['multi_news']}",
    ]

    document = f" {util.DOC_SEP_TOKENS['multi_news']} ".join(docs)
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
    ref_abstract = {"abstract": ["This is a cited abstract."]}
    related_work = "This is the related work."

    expected_text, expected_summary = (
        "This is the query abstract. <doc-sep> This is a cited abstract.",
        "This is the related work.",
    )
    actual_text, actual_summary = util.preprocess_multi_x_science_sum(
        text=abstract,
        summary=related_work,
        ref_abstract=ref_abstract,
        doc_sep_token=doc_sep_token,
    )
    assert expected_text == actual_text
    assert expected_summary == actual_summary


def test_preprocess_ms2() -> None:
    doc_sep_token = "<doc-sep>"

    background = "This is the background."
    titles = ["This is title 1.", "This is title 2."]
    abstracts = ["This is abstract 1.", "This is abstract 2."]
    summary = "This is the summary."

    expected_text, expected_summary = (
        (
            "This is the background. <doc-sep> This is title 1. This is abstract 1. <doc-sep> This is title 2."
            " This is abstract 2."
        ),
        "This is the summary.",
    )

    actual_text, actual_summary = util.preprocess_ms2(
        text=background,
        summary=summary,
        titles=titles,
        abstracts=abstracts,
        doc_sep_token=doc_sep_token,
    )
    assert expected_text == actual_text
    assert expected_summary == actual_summary
