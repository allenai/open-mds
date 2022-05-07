from typing import Callable
import pytest
from retrieval_exploration.common import util
from transformers import PreTrainedTokenizer


def test_preprocess_multi_news() -> None:
    # Could be anything, choose the PRIMERA document separator token
    doc_sep_token = "<doc-sep>"
    text = "Document numero uno {util._DOC_SEP_TOKENS['multi_news']} Document numero dos"
    summary = "This can be anything"
    expected_text, expected_summary = (
        text.replace(util._DOC_SEP_TOKENS["multi_news"], "<doc-sep>"),
        summary,
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


def test_get_global_attention_mask():
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
