from typing import List, Tuple
from transformers import PreTrainedTokenizer

_DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multi_news": "|||||"}


def preprocess_multi_news(text: str, summary: str, doc_sep_token: str) -> Tuple[str, str]:
    # Some examples have a doc sep token at the end of the text, so strip it before we split.
    text = text.strip(_DOC_SEP_TOKENS["multi_news"]).strip()
    text = text.replace(_DOC_SEP_TOKENS["multi_news"], doc_sep_token)
    return text, summary


def get_doc_sep_token(tokenizer: PreTrainedTokenizer) -> str:
    """Returns a suitable document seperator token depending on `tokenizer`. In general, the
    function checks if this `tokenizer.name_or_path` has a special document token (defined in
    `common.util._DOC_SEP_TOKENS`). If that is not found, it then checks for: `tokenizer.sep_token`,
    `tokenizer.bos_token`, `tokenizer.eos_token` in that order. If these are all `None`, a
    `ValueError` is raised.
    """
    # PRIMERA models have their own special token, <doc-sep>.
    if "primera" in tokenizer.name_or_path.lower():
        return _DOC_SEP_TOKENS["primera"]
    elif tokenizer.sep_token is not None:
        return tokenizer.sep_token
    elif tokenizer.bos_token is not None:
        return tokenizer.bos_token
    elif tokenizer.eos_token is not None:
        return tokenizer.eos_token
    else:
        raise ValueError(
            f"Could not determine a suitable document sperator token '{tokenizer.name_or_path}'"
        )


def truncate_multi_doc(
    text: str, doc_sep_token: str, max_length: int, tokenizer: PreTrainedTokenizer
) -> str:
    """Given some `text`, which is assumed to be multiple documents joined by `doc_sep_token`,
    truncates each document (using `tokenizer`) so that the length of the concatenation of all
    documents does not exceed max_length. See https://aclanthology.org/2021.naacl-main.380/ and
    https://arxiv.org/abs/2110.08499 for more details.
    """
    # Some datasets have the doc sep token at the end of the text, so strip it before we split.
    docs = text.strip(doc_sep_token).strip().split(doc_sep_token)
    # -2 to make room for the special tokens, -(len(docs) - 1) to make room for the doc sep tokens.
    max_doc_length = (max_length - 2 - (len(docs) - 1)) // len(docs)
    truncated_docs = []
    for doc in docs:
        # Truncate each doc to its maximum allowed length
        truncated_docs.append(
            tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(doc, max_length=max_doc_length, truncation=True)
                # Going to join everything on a space at the end, so strip it off here.
            ).strip()
        )
    return f" {doc_sep_token} ".join(truncated_docs)


def get_global_attention_mask(input_ids: List[List[int]], token_ids: List[int]) -> List[List[int]]:
    """Returns a corresponding global attention mask for `input_ids`, which is 1 for any tokens in
    `token_ids` (indicating the model should attend to those tokens) and 0 elsewhere (indicating the
    model should not attend to those tokens).

    # Parameters

    input_ids : `List[List[str]]`
        The input ids that will be provided to a model during the forward pass.
    token_ids : `List[List[str]]`
        The token ids that should be globally attended to.
    """

    # TODO (John): Ideally this would be vectorized
    global_attention_mask = [
        [1 if token_id in token_ids else 0 for token_id in batch] for batch in input_ids
    ]
    return global_attention_mask
