from typing import List

_DOC_SEP_TOKENS = {"primera": "<doc-sep>", "multinews": "|||||"}


def preprocess_multi_news(text: str, summary: str, doc_sep_token: str) -> str:
    return text.replace(_DOC_SEP_TOKENS["multinews"], doc_sep_token), summary


def get_doc_sep_token(model_name_or_path, tokenizer) -> str:
    """Returns a suitable document seperator token depending on `model_name_or_path` and
    `tokenizer`. In general, the function checks if this `model_name_or_path` has a special
    document token (defined in `common.util._DOC_SEP_TOKENS`). If that is not found, it then checks
    for: `tokenizer.sep_token`, `tokenizer.bos_token`, `tokenizer.eos_token` (in that order). If
    these are all `None`, a `ValueError` is raised.
    """
    # PRIMERA models have their own special token, <doc-sep>.
    if "primera" in model_name_or_path.lower():
        return _DOC_SEP_TOKENS["primera"]
    elif tokenizer.sep_token is not None:
        return tokenizer.sep_token
    elif tokenizer.bos_token is not None:
        return tokenizer.bos_token
    elif tokenizer.eos_token is not None:
        return tokenizer.eos_token
    else:
        raise ValueError(
            f"Could not determine a suitable document sperator token for mode '{model_name_or_path}'"
        )


def get_global_attention_mask(input_ids: List[List[str]], token_ids: List[int]) -> List[List[int]]:
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
        [1 if token_id in token_ids else 0 for token_id in batch]
        for batch in input_ids
    ]
    return global_attention_mask
