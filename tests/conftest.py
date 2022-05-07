from typing import Callable
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer


@pytest.fixture
def hf_tokenizer() -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create an AllenNLP `Vocabulary` object. It accepts optional `**extras`
    which will be used along with `params` to create the `Vocabulary` object.
    """

    def _hf_tokenizer(model_name_or_path: str, **kwargs) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

    return _hf_tokenizer
