#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import flatten_dict
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import transformers
from datasets import load_dataset, load_from_disk, load_metric
from filelock import FileLock
from retrieval_exploration.common import util
from retrieval_exploration.perturbations import Perturber
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.21.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This "
                "option should only be set to `True` for repositories you trust and in which you have read the "
                "code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class PerturbationArguments:
    """
    Arguments pertaining to the perturbations we will apply to the source documents.
    """

    perturbation: Optional[str] = field(
        default=None,
        metadata={"help": "The pertubation to apply."},
    )
    selection_strategy: str = field(
        default="random",
        metadata={
            "help": "The selection strategy to use for the perturbation. Has no effect if perturbation is None."
        },
    )
    perturbed_frac: Optional[float] = field(
        default=None,
        metadata={"help": "Percent of input documents to perturb. Has no effect if perturbation is None."},
    )
    perturbed_seed: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Random seed that will be set locally when perturbing inputs.",
                " Has no effect if perturbation is None.",
            )
        },
    )


@dataclass
class RetrievalArguments:
    """
    Arguments pertaining to the retrieval experiements. Have no effect, but will be recorded in the output.
    """

    retriever: Optional[str] = field(
        default=None,
        metadata={"help": "The type of retrieval pipeline to use."},
    )
    top_k_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The strategy to use when choosing the k top documents to retrieve. If 'oracle' (default), k is"
                " chosen as the number of source documents in the original example. If 'max', k is chosen as the"
                " maximum number of source documents across the examples of the dataset. If 'mean', k is chosen as"
                " the mean number of source documents across the examples of the dataset."
            ),
        },
    )


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, PerturbationArguments, RetrievalArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, perturbation_args, retrieval_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    # Our unique parsing strategy (which depends on OmegaConf) exists here
    elif any(argv.endswith(".yml") for argv in sys.argv[1:]):
        conf = util.parse_omega_conf()
        model_args, data_args, training_args, perturbation_args, retrieval_args = parser.parse_dict(conf)
    else:
        model_args, data_args, training_args, perturbation_args, retrieval_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    #  Loading a dataset from a local directory.
    if Path(data_args.dataset_name).is_dir():
        raw_datasets = load_from_disk(
            data_args.dataset_name,
        )
    # Downloading and loading a dataset from the hub.
    elif data_args.dataset_name is not None:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def _preprocess_batch(examples) -> Dict[str, List[str]]:
        """Handles the logic for preprocessing a batch of examples."""

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            # remove pairs where at least one record is None
            if examples[text_column][i]:
                if data_args.dataset_name == "multi_news" or "multinews" in data_args.dataset_name:
                    text, summary = util.preprocess_multi_news(
                        text=examples[text_column][i],
                        summary=examples[summary_column][i],
                        doc_sep_token=doc_sep_token,
                    )
                elif data_args.dataset_name == "ccdv/WCEP-10" or "wcep" in data_args.dataset_name:
                    text, summary = util.preprocess_wcep(
                        text=examples[text_column][i],
                        summary=examples[summary_column][i],
                        doc_sep_token=doc_sep_token,
                    )
                elif data_args.dataset_name == "multi_x_science_sum" or "multixscience" in data_args.dataset_name:
                    text, summary = util.preprocess_multi_x_science_sum(
                        text=examples[text_column][i],
                        summary=examples[summary_column][i],
                        ref_abstract=examples["ref_abstract"][i],
                        doc_sep_token=doc_sep_token,
                    )
                elif data_args.dataset_config_name == "ms2" or "ms2" in data_args.dataset_name:
                    text, summary = util.preprocess_ms2(
                        text=examples[text_column][i],
                        summary=examples[summary_column][i],
                        titles=examples["title"][i],
                        abstracts=examples["abstract"][i],
                        doc_sep_token=doc_sep_token,
                    )
                elif data_args.dataset_config_name == "cochrane" or "cochrane" in data_args.dataset_name:
                    text, summary = util.preprocess_cochrane(
                        summary=examples[summary_column][i],
                        titles=examples["title"][i],
                        abstracts=examples[text_column][i],
                        doc_sep_token=doc_sep_token,
                    )
                else:
                    text, summary = examples[text_column][i], examples[summary_column][i]

                # Do some basic cleanup on the source text
                text = util.sanitize_text(text)

                inputs.append(text)
                targets.append(summary)

        inputs = [prefix + inp for inp in inputs]

        return {"inputs": inputs, "targets": targets}

    def preprocess_function(examples):
        preprocessed_batch = _preprocess_batch(examples)
        inputs, targets = preprocessed_batch["inputs"], preprocessed_batch["targets"]

        # Before we perturb...
        # record the number of documents in each instance
        orig_num_docs = [util.get_num_docs(text, doc_sep_token=doc_sep_token) for text in inputs]

        if perturbation_args.perturbation is None:
            logger.info("No perturbations will be applied.")
        else:
            perturber = Perturber(
                perturbation_args.perturbation,
                doc_sep_token=doc_sep_token,
                strategy=perturbation_args.selection_strategy,
                seed=perturbation_args.perturbed_seed,
            )

            unperturbed_indices = None
            # Both Multi-XScience and MS^2 examples begin with a "special" document, the abstract of the paper
            # whos related works section we are trying to generate, and the background section of the literature
            # review we are trying to generate, respectively. Both of these should be excluded from perturbation,
            # as they are not something we would retrieve.
            if (
                data_args.dataset_name == "multi_x_science_sum"
                or data_args.dataset_config_name == "ms2"
                or "ms2" in data_args.dataset_name
            ):
                unperturbed_indices = [0]
                logger.info(f"Documents at indices '{unperturbed_indices}' will not be perturbed.")

            inputs = perturber(
                inputs,
                perturbed_frac=perturbation_args.perturbed_frac,
                targets=targets,
                documents=documents,
                unperturbed_indices=unperturbed_indices,
            )
            logger.info(
                f"Applying perturbation '{perturbation_args.perturbation}' with selection strategy '{perturbation_args.selection_strategy}'."
            )

        # Rather than naively truncating the concatenated documents, we follow
        # https://aclanthology.org/2021.naacl-main.380/ and https://arxiv.org/abs/2110.08499
        # by truncating each document separately to statisfy the max length of the input.
        # We need to be careful that we control for this truncation during our perturbation
        # experiments, so we compute the number of original documents in the unperturbed
        # input and use that to determine the allowed length of each input document.
        inputs = [
            util.truncate_multi_doc(
                text,
                doc_sep_token=doc_sep_token,
                max_length=data_args.max_source_length,
                tokenizer=tokenizer,
                num_docs=num_docs,
            )
            for text, num_docs in zip(inputs, orig_num_docs)
        ]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # Add a global attention mask to models inputs. We don't bother checking if the model will
        # actually use it, as it will be ignored if not. For summarization, we place global attention
        # on the document seperator token and the bos token (if it exists).
        global_attention_tokens = [doc_sep_token]
        if tokenizer.bos_token is not None:
            global_attention_tokens.append(tokenizer.bos_token)
        model_inputs["global_attention_mask"] = util.get_global_attention_mask(
            model_inputs.input_ids,
            token_ids=tokenizer.convert_tokens_to_ids(global_attention_tokens),
        )
        logger.info(f"Using global attention on the following tokens: {global_attention_tokens}")
        return model_inputs

    # Determine the document seperator token for this model.
    doc_sep_token = util.get_doc_sep_token(tokenizer)
    logger.info(f"Using {doc_sep_token} as the document seperator token.")

    # Certain perturbations require us to provide all examples in the dataset, collect those here
    documents = None
    if perturbation_args.perturbation in ["addition", "replacement"] and perturbation_args.perturbed_frac:
        documents = []
        for split in raw_datasets:
            preprocessed_dataset = raw_datasets[split].map(
                _preprocess_batch,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Collecting documents from {split} set",
            )
            documents.extend(preprocessed_dataset["inputs"])
        logger.info(f"Loaded {len(documents)} documents. These will be included for selection during perturbation.")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
                batch_size=4096,
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
                batch_size=None,
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
                batch_size=None,
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metrics
    rouge = load_metric("rouge")
    bertscore = load_metric("bertscore")

    def postprocess_text(preds, labels):
        # Clean text by removing whitespace, newlines and tabs
        preds = [" ".join(pred.strip().split()) for pred in preds]
        labels = [" ".join(label.strip().split()) for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels, inputs = eval_preds.predictions, eval_preds.label_ids, eval_preds.inputs
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if inputs is not None:
                inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        # Compute and post-process rouge results
        rouge_results = rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            use_aggregator=False,
        )
        for key, value in rouge_results.items():
            rouge_results[key] = {
                "precision": [score.precision * 100 for score in value],
                "recall": [score.recall * 100 for score in value],
                "fmeasure": [score.fmeasure * 100 for score in value],
                "fmeasure_mean": np.mean([score.fmeasure for score in value]) * 100,
            }
        # Compute the arithmetic mean of ROUGE-1, ROUGE-2 and ROUGE-L following: https://arxiv.org/abs/2110.08499
        rouge_results["rouge_avg_fmeasure"] = np.mean(
            [rouge_results[key]["fmeasure"] for key in ["rouge1", "rouge2", "rougeL"]], axis=0
        ).tolist()
        rouge_results["rouge_avg_fmeasure_mean"] = np.mean(rouge_results["rouge_avg_fmeasure"]).item()

        # Compute and post-process bertscore results
        bertscore_results = bertscore.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            # These are mostly based on the recommendations in https://github.com/Tiiiger/bert_score
            model_type="microsoft/deberta-xlarge-mnli",
            # We can generally afford to use a batch size 4X greater than the eval batch size
            batch_size=training_args.per_device_eval_batch_size * 4,
            device=training_args.device,
            lang="en",
            rescale_with_baseline=True,
            use_fast_tokenizer=True,
        )
        bertscore_results["f1_mean"] = np.mean(bertscore_results["f1"])
        for key, value in bertscore_results.items():
            if key == "hashcode":
                continue
            if isinstance(value, list):
                bertscore_results[key] = [score * 100 for score in value]
            else:
                bertscore_results[key] = value * 100

        # Collect results in final (flat) dict
        results = {
            **flatten_dict.flatten(rouge_results, reducer="underscore"),
            **flatten_dict.flatten({"bertscore": bertscore_results}, reducer="underscore"),
        }

        # Log some additional, split-dependent information in the results dictionary

        # Perturbations
        results["perturbation"] = perturbation_args.perturbation
        results["selection_strategy"] = perturbation_args.selection_strategy
        results["perturbed_frac"] = perturbation_args.perturbed_frac
        results["perturbed_seed"] = perturbation_args.perturbed_seed
        # Retrieval
        results["retriever"] = retrieval_args.retriever
        results["top_k_strategy"] = retrieval_args.top_k_strategy
        # I/O
        results["labels"] = decoded_labels
        results["preds"] = decoded_preds
        if inputs is not None:
            results["inputs"] = util.batch_decode_multi_doc(inputs, tokenizer, doc_sep_token=doc_sep_token)
        # Add length of reference and generated summaries
        results["label_len"] = [np.count_nonzero(label != tokenizer.pad_token_id) for label in labels]
        results["pred_len"] = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        results["input_len"] = [np.count_nonzero(example != tokenizer.pad_token_id) for example in inputs]
        return results

    # Log some additional, split-agnositic information in the all_results dictionary
    metadata = {
        "seed": training_args.seed,
        "doc_sep_token": doc_sep_token,
        "model_name_or_path": model_args.model_name_or_path,
    }

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics.update(metadata)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        metrics.update(metadata)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        metrics.update(metadata)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                # Predictions may contain "\n", which causes this file to contain an incorrect number of lines.
                # Pointed this out to HF maintainers here: https://github.com/huggingface/transformers/issues/18992
                # but they weren't interested in fixing it.
                predictions = [util.sanitize_text(pred) for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
