from typing import Callable
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer, AutoModel, AutoConfig
import datasets
from datasets.dataset_dict import DatasetDict
from transformers import PretrainedConfig


@pytest.fixture
def hf_dataset() -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create a HuggingFace dataset object. Optional **kwargs are passed to `load_dataset()`.
    """

    def _hf_dataset(model_name_or_path: str, **kwargs) -> DatasetDict:
        return datasets.load_dataset(model_name_or_path, **kwargs)

    return _hf_dataset


@pytest.fixture
def hf_tokenizer() -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create a HuggingFace tokenizer object. Optional **kwargs are passed to `from_pretrained()`.
    """

    def _hf_tokenizer(model_name_or_path: str, **kwargs) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)

    return _hf_tokenizer


@pytest.fixture
def hf_config() -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create a HuggingFace config object. Optional **kwargs are passed to `from_pretrained()`.
    """

    def _hf_config(model_name_or_path: str, **kwargs) -> PretrainedConfig:
        return AutoConfig.from_pretrained(model_name_or_path, **kwargs)

    return _hf_config


@pytest.fixture
def hf_model() -> Callable:
    """This is a fixture factory. It returns a function that you can use
    to create a HuggingFace model object. Optional **kwargs are passed to `from_pretrained()`.
    """

    def _hf_model(model_name_or_path: str, **kwargs):
        return AutoModel.from_pretrained(model_name_or_path, **kwargs)

    return _hf_model
