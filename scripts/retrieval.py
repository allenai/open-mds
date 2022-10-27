import copy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List

import pyterrier as pt
import typer
from retrieval_exploration import indexing
from retrieval_exploration.common import util
from rich import print

app = typer.Typer()

# This controls the sizes of batches which are written to disk by HuggingFace Datasets.
# Smaller values save memory at the cost of more compute
_WRITER_BATCH_SIZE = 500

# The default location to save document indices.
_DOCUMENT_INDEX_DIR = Path(util.CACHE_DIR) / "indices"

# The neural retirever to use for dense retireval pipeline. This could be made an argument to the script.
_DEFAULT_NEURAL_RETRIEVER = "facebook/contriever-msmarco"


class HFDatasets(str, Enum):
    multinews = "multinews"
    wcep = "wcep"
    multixscience = "multixscience"
    ms2 = "ms2"
    cochrane = "cochrane"


class Retriever(str, Enum):
    sparse = "sparse"
    dense = "dense"


class TopKStrategy(str, Enum):
    mean = "mean"
    max_ = "max"
    oracle = "oracle"


@app.command()
def main(
    hf_dataset_name: HFDatasets = typer.Argument(
        ..., case_sensitive=False, help="The name of a supported HuggingFace Dataset."
    ),
    output_dir: Path = typer.Argument(
        ...,
        help=("Path to the directory where the dataset and retrieval results will be saved."),
    ),
    index_path: Path = typer.Option(
        None,
        help=(
            "Directory to save the PyTerrier index. If an index already exists at this path and"
            " --overwrite-index is not passed, the index will be overwritten. If not provided, the index will be"
            " saved to util.CACHE_DIR / 'indexes'."
        ),
    ),
    retriever: Retriever = typer.Option(
        Retriever.sparse, case_sensitive=False, help="The type of retrieval pipeline to use."
    ),
    model_name_or_path: str = typer.Option(
        _DEFAULT_NEURAL_RETRIEVER,
        help=(
            "Which model to use for dense retrieval. Can be any Sentence Transformer or HuggingFace Transformer"
            f" model. Defaults to {_DEFAULT_NEURAL_RETRIEVER} Has no effect if choosen retriever does not use a"
            " neural model."
        ),
    ),
    top_k_strategy: TopKStrategy = typer.Option(
        TopKStrategy.oracle,
        case_sensitive=False,
        help=(
            "The strategy to use when choosing the k top documents to retrieve. If 'oracle' (default), k is"
            " chosen as the number of source documents in the original example. If 'max', k is chosen as the"
            " maximum number of source documents across the examples of the dataset. If 'mean', k is chosen as the"
            " mean number of source documents across the examples of the dataset."
        ),
    ),
    splits: List[str] = typer.Option(
        None, help="Which splits of the dataset to replace with retrieved documents. Defaults to all splits."
    ),
    overwrite_index: bool = typer.Option(
        False, "--overwrite-index", help="Overwrite the PyTerrier index at --index-path, if it exists."
    ),
    overwrite_cache: bool = typer.Option(
        False, "--overwrite-cache", help="Overwrite the cached copy of the HuggingFace dataset, if it exits."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Perform retrieval and report results without re-building the dataset. Useful for tuning and evaluation.",
    ),
) -> None:
    """Recreates the chosen HuggingFace dataset using the documents retrieved from an IR system."""

    # Any dataset specific setup goes here
    if hf_dataset_name == HFDatasets.multinews:
        path = "multi_news"
        doc_sep_token = util.DOC_SEP_TOKENS[path]
        pt_dataset = indexing.CanonicalMDSDataset(path, doc_sep_token=doc_sep_token)
    elif hf_dataset_name == HFDatasets.wcep:
        path = "ccdv/WCEP-10"
        doc_sep_token = util.DOC_SEP_TOKENS[path]
        pt_dataset = indexing.CanonicalMDSDataset(path, doc_sep_token=doc_sep_token)
    elif hf_dataset_name == HFDatasets.multixscience:
        pt_dataset = indexing.MultiXScienceDataset()
    elif hf_dataset_name == HFDatasets.ms2 or hf_dataset_name == HFDatasets.cochrane:
        pt_dataset = indexing.MSLR2022Dataset(name=hf_dataset_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Create a directory to store the index if it wasn't provided
    index_path = Path(index_path) if index_path is not None else _DOCUMENT_INDEX_DIR / pt_dataset.path
    if pt_dataset.name is not None:
        index_path = index_path / pt_dataset.name
    index_path.mkdir(parents=True, exist_ok=True)

    # Use all splits if not specified
    splits = splits or list(pt_dataset._hf_dataset.keys())
    print(f"[bold blue]:information: Will replace documents in {', '.join(splits)} splits.")

    # Create a new copy of the dataset and replace its source documents with retrieved documents
    hf_dataset = copy.deepcopy(pt_dataset._hf_dataset)
    print(f"[bold green]:white_check_mark: Loaded the dataset from '{pt_dataset.info_url()}' [/bold green]")

    if retriever == Retriever.sparse:
        indexref = pt_dataset.get_index(str(index_path), overwrite=overwrite_index, verbose=True)
        # In general, we should always load the actual index
        # See: https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html#index-like-objects
        index = pt.IndexFactory.of(indexref)
        print(f"[bold green]:white_check_mark: Loaded the index from '{index_path}' [/bold green]")
        retrieval_pipeline = pt.BatchRetrieve(index, wmodel="BM25", metadata=["docno", "text"], verbose=True)
    else:
        # Import here as PyTerrier will have been initialized by this point
        from pyterrier_sentence_transformers import SentenceTransformersIndexer, SentenceTransformersRetriever

        indexer = SentenceTransformersIndexer(
            model_name_or_path=model_name_or_path,
            index_path=str(index_path),
            overwrite=overwrite_index,
            normalize=False,
            verbose=False,
        )
        indexer.index(pt_dataset.get_corpus_iter(verbose=True))
        print(f"[bold green]:white_check_mark: Loaded the index from '{index_path}' [/bold green]")
        retrieval_pipeline = SentenceTransformersRetriever(
            model_name_or_path=model_name_or_path, index_path=str(index_path), verbose=False
        )
    print(f"[bold green]:white_check_mark: Loaded the '{retriever.value}' retrieval pipeline[/bold green]")

    top_k_strategy_msg = f"[bold blue]:hammer_and_wrench: Using the '{top_k_strategy.value}' TopKStrategy. "
    if top_k_strategy.value != TopKStrategy.oracle:
        # Following https://arxiv.org/abs/2104.06486, take the first 25 articles
        if hf_dataset_name == HFDatasets.ms2 or hf_dataset_name == HFDatasets.cochrane:
            document_stats = pt_dataset.get_document_stats(max_included_studies=25)
        else:
            document_stats = pt_dataset.get_document_stats()
        k = int(round(document_stats[top_k_strategy.value], 0))
        print(top_k_strategy_msg + f"k will be set statically to {k} [/bold blue]")
    else:
        k = None
        print(
            top_k_strategy_msg
            + "k will be set dynamically as the original number of documents in each example [/bold blue]"
        )

    for split in splits:
        # Use PyTerrier to actually perform the retrieval and then replace the source docs with the retrieved docs
        # See: https://pyterrier.readthedocs.io/en/latest/terrier-retrieval.html
        topics = pt_dataset.get_topics(split)
        qrels = pt_dataset.get_qrels(split)
        retrieved = retrieval_pipeline.transform(topics)

        print(f"[bold blue]:test_tube: Evaluating '{retriever}' retrieval pipeline on the '{split}' set [/bold blue]")
        print(
            pt.Experiment(
                [retrieved],
                topics=topics,
                qrels=qrels,
                eval_metrics=["ndcg", "recall_100", "recall_1000", "Rprec"],
                names=[retriever.value],
                save_dir=output_dir,
                save_mode="overwrite",
                round=4,
            )
        )

        if dry_run:
            print("[bold yellow]:warning: --dry-run argument provided, dataset will not be re-built[/bold yellow]")
            continue

        hf_dataset[split] = hf_dataset[split].map(
            partial(pt_dataset.replace, split=split, retrieved=retrieved, k=k),
            with_indices=True,
            load_from_cache_file=not overwrite_cache,
            writer_batch_size=_WRITER_BATCH_SIZE,
            desc=f"Re-building {split} set",
        )
        print(f"[bold blue]:repeat: Source documents in '{split}' set replaced with retrieved documents[/bold blue]")

    if not dry_run:
        hf_dataset.save_to_disk(output_dir)
        print(f"[bold green]:floppy_disk: Re-built dataset saved to {output_dir} [/bold green]")


if __name__ == "__main__":
    app()
