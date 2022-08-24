import shutil
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
import pyterrier as pt
from datasets import load_dataset
from tqdm import tqdm

from retrieval_exploration.common import util

_HF_DATASETS_URL = "https://huggingface.co/datasets"

if not pt.started():
    # Required if you want to use pyterrier offline
    pt.init(version=5.6, helper_version="0.0.6")
    # pt.init()


def _get_iter_dict_indexer(index_path: str, dataset: pt.datasets.Dataset, **kwargs) -> pt.IterDictIndexer:
    """Helper function that returns a PyTerrier IterDictIndexer with the correct meta index fields and lengths."""
    docno_lengths, text_lengths = zip(
        *[(len(item["docno"]), len(item["text"])) for item in dataset.get_corpus_iter(verbose=False)]
    )
    max_docno_length = max(docno_lengths)
    # Take the 95th percentile of lengths to avoid blowing up the meta index size.
    max_text_length = int(np.percentile(text_lengths, q=95))

    # Store text as an attribute, which is required by some transforms
    # See: https://pyterrier.readthedocs.io/en/latest/text.html
    return pt.IterDictIndexer(index_path, meta={"docno": max_docno_length, "text": max_text_length}, **kwargs)


def _sanitize_query(topics: pd.DataFrame) -> pd.DataFrame:
    """Helper function that strips markup tokens from a query that would otherwise cause errors with PyTerrier.
    See: https://github.com/terrier-org/pyterrier/issues/253#issuecomment-996160987
    """

    def _strip_markup(text: str) -> str:
        tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
        return " ".join(tokenizer.getTokens(text))

    return pt.apply.query(lambda x: _strip_markup(x.query))(topics)


class HuggingFacePyTerrierDataset(pt.datasets.Dataset):
    """Simple wrapper for the PyTerrier Dataset class to make it easier to interface with HuggingFace Datasets."""

    def __init__(self, path: str, name: Optional[str] = None, **kwargs) -> None:
        self.path = path
        self.name = name
        self._hf_dataset = load_dataset(self.path, self.name, **kwargs)

    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, retrieved: pd.DataFrame, k: Optional[int] = None
    ) -> Dict[str, Any]:
        """This method replaces the original source documents of an `example` from a HuggingFace dataset with the
        top-`k` documents in `retrieved`. It is expected that this function will be passed to the `map` method of
        the HuggingFace Datasets library with the argument `with_indices=True`. Must be implemented by child class.
        If `k` is `None`, it will be set dynamically for each example as the original number of source documents.
        """
        raise NotImplementedError("Static method 'replace' must be implemented by the child class.")

    def get_index(self, index_path: str, overwrite: bool = False, verbose: bool = True, **kwargs) -> pt.IndexRef:
        """Returns the `IndexRef` for this dataset from `index_path`, creating it first if it doesn't already
        exist. If `overwrite`, the index will be rebuilt. Any provided **kwargs are passed to `pt.IterDictIndexer`.
        """
        if any(Path(index_path).iterdir()):
            if overwrite:
                shutil.rmtree(index_path)
            else:
                return pt.IndexRef.of(index_path)

        indexer = _get_iter_dict_indexer(index_path, dataset=self, **kwargs)

        # Compose index from iterator
        # See: https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html#iterdictindexer
        indexref = indexer.index(self.get_corpus_iter(verbose=verbose))
        return indexref

    def info_url(self) -> str:
        return f"{_HF_DATASETS_URL}/{self.path}"


class MultiNewsDataset(HuggingFacePyTerrierDataset):
    def __init__(self, **kwargs):
        super().__init__("multi_news", None, **kwargs)

    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, retrieved: pd.DataFrame, k: Optional[int] = None
    ) -> Dict[str, Any]:
        # Multi-News has a special document seperator token that we need to parse out individual documents
        doc_sep_token = util.DOC_SEP_TOKENS["multi_news"]
        qid = f"{split}_{idx}"
        k = k or util.get_num_docs(example["document"], doc_sep_token=doc_sep_token)
        # We would like to get the original, unaltered text from the dataset, so we use the docno's to key in.
        # It would be less complicated to retrieve the text from the PyTerrier MetaIndex, but these documents are
        # not identical due to some string processing.
        retrieved_docs = []
        retrieved_docnos = retrieved[retrieved.qid == qid][:k]["docno"].tolist()
        for docno in retrieved_docnos:
            retrieved_split, example_idx, document_idx = docno.split("_")
            retrieved_example = self._hf_dataset[retrieved_split]["document"][int(example_idx)]
            retrieved_doc = util.split_docs(retrieved_example, doc_sep_token=doc_sep_token)[int(document_idx)]
            retrieved_docs.append(retrieved_doc)
        example["document"] = f" {doc_sep_token} ".join(doc.strip() for doc in retrieved_docs)
        return example

    def get_corpus_iter(self, verbose: bool = True) -> Iterator[Dict[str, Any]]:
        yielded = set()
        for split in self._hf_dataset:
            for i, example in tqdm(
                enumerate(self._hf_dataset[split]),
                desc=f"Indexing {split}",
                total=len(self._hf_dataset[split]),
                disable=not verbose,
            ):
                docs = util.split_docs(example["document"], doc_sep_token=util.DOC_SEP_TOKENS[self.path])
                for j, doc in enumerate(docs):
                    doc = doc.strip()
                    # Don't index duplicate or empty documents
                    if doc in yielded or not doc:
                        continue
                    yielded.add(doc)
                    # These documents don't have unique IDs, so create them using the split name and index
                    yield {"docno": f"{split}_{i}_{j}", "text": doc}

    def get_topics(self, split: str, max_examples: Optional[int] = None) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        if max_examples:
            dataset = dataset[:max_examples]
        queries = dataset["summary"]
        qids = [f"{split}_{i}" for i in range(len(queries))]
        topics = pd.DataFrame({"qid": qids, "query": queries})
        return _sanitize_query(topics)

    def get_qrels(self, split: str) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        qids, docnos = [], []
        for i, example in enumerate(dataset):
            docs = util.split_docs(example["document"], doc_sep_token=util.DOC_SEP_TOKENS[self.path])
            for j, _ in enumerate(docs):
                qids.append(f"{split}_{i}")
                docnos.append(f"{split}_{i}_{j}")
        labels = [1] * len(qids)
        return pd.DataFrame({"qid": qids, "docno": docnos, "label": labels})

    def get_document_stats(self) -> Dict[str, float]:
        num_docs = []
        for split in self._hf_dataset:
            for example in self._hf_dataset[split]:
                num_docs.append(util.get_num_docs(example["document"], doc_sep_token=util.DOC_SEP_TOKENS[self.path]))
        return {"max": np.max(num_docs), "mean": np.mean(num_docs), "min": np.min(num_docs)}


class MultiXScienceDataset(HuggingFacePyTerrierDataset):
    def __init__(self, **kwargs):
        super().__init__("multi_x_science_sum", None, **kwargs)

    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, retrieved: pd.DataFrame, k: Optional[int] = None
    ) -> Dict[str, Any]:
        qid = f"{split}_{idx}"
        k = k or len(example["ref_abstract"]["abstract"])
        # We would like to get the original, unaltered text from the dataset, so we use the docno's to key in.
        # It would be less complicated to retrieve the text from the PyTerrier MetaIndex, but these documents are
        # not identical due to some string processing.
        retrieved_docs = []
        retrieved_docnos = retrieved[retrieved.qid == qid][:k]["docno"].tolist()
        for docno in retrieved_docnos:
            retrieved_split, example_idx, document_idx = docno.split("_")
            retrieved_doc = self._hf_dataset[retrieved_split][int(example_idx)]["ref_abstract"]["abstract"][
                int(document_idx)
            ]
            retrieved_docs.append(retrieved_doc)
        example["ref_abstract"]["abstract"] = [doc for doc in retrieved_docs]
        return example

    def get_corpus_iter(self, verbose: bool = True) -> Iterator[Dict[str, Any]]:
        yielded = set()
        for split in self._hf_dataset:
            for i, example in tqdm(
                enumerate(self._hf_dataset[split]),
                desc=f"Indexing {split}",
                total=len(self._hf_dataset[split]),
                disable=not verbose,
            ):
                for j, doc in enumerate(example["ref_abstract"]["abstract"]):
                    doc = doc.strip()
                    # Don't index duplicate or empty documents
                    if doc in yielded or not doc:
                        continue
                    yielded.add(doc)
                    # These documents don't have unique IDs, so create them using the split name and index
                    yield {"docno": f"{split}_{i}_{j}", "text": doc}

    def get_topics(self, split: str, max_examples: Optional[int] = None) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        if max_examples:
            dataset = dataset[:max_examples]
        queries = [f"{related_work.strip()}" for related_work in dataset["related_work"]]
        qids = [f"{split}_{i}" for i in range(len(queries))]
        topics = pd.DataFrame({"qid": qids, "query": queries})
        return _sanitize_query(topics)

    def get_qrels(self, split: str) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        qids, docnos = [], []
        for i, example in enumerate(dataset):
            for j, _ in enumerate(example["ref_abstract"]["abstract"]):
                qids.append(f"{split}_{i}")
                docnos.append(f"{split}_{i}_{j}")
        labels = [1] * len(qids)
        return pd.DataFrame({"qid": qids, "docno": docnos, "label": labels})

    def get_document_stats(self) -> Dict[str, float]:
        num_docs = []
        for split in self._hf_dataset:
            for example in self._hf_dataset[split]:
                num_docs.append(len(example["ref_abstract"]["abstract"]))
        return {"max": np.max(num_docs), "mean": np.mean(num_docs), "min": np.min(num_docs)}


class MS2Dataset(HuggingFacePyTerrierDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__("allenai/mslr2022", "ms2", **kwargs)

    def replace(
        self, example: Dict[str, Any], idx: int, *, split: str, retrieved: pd.DataFrame, k: Optional[int] = None
    ) -> Dict[str, Any]:
        qid = example["review_id"]
        k = k or len(example["pmid"])
        # We would like to get the original, unaltered text from the dataset, so we use the docno's to key in.
        # It would be less complicated to retrieve the text from the PyTerrier MetaIndex, but these documents are
        # not identical due to some string processing.
        retrieved_titles, retrieved_abstracts, retrieved_pmids = [], [], []
        retrieved_docnos = retrieved[retrieved.qid == qid][:k]["docno"].tolist()
        for docno in retrieved_docnos:
            retrieved_split, review_id, pmid = docno.split("_")
            # Convert these strings to indices for easier lookup
            example_idx = self._hf_dataset[retrieved_split]["review_id"].index(review_id)
            retrieved_example = self._hf_dataset[retrieved_split][example_idx]
            document_idx = retrieved_example["pmid"].index(pmid)
            retrieved_titles.append(retrieved_example["title"][document_idx])
            retrieved_abstracts.append(retrieved_example["abstract"][document_idx])
            retrieved_pmids.append(pmid)
        example["title"] = [title for title in retrieved_titles]
        example["abstract"] = [abstract for abstract in retrieved_abstracts]
        example["pmid"] = [pmid for pmid in retrieved_pmids]
        return example

    def get_corpus_iter(self, verbose: bool = False):
        yielded = set()
        for split in self._hf_dataset:
            for example in tqdm(
                self._hf_dataset[split],
                desc=f"Indexing {split}",
                total=len(self._hf_dataset[split]),
                disable=not verbose,
            ):
                for title, abstract, pmid in zip(example["title"], example["abstract"], example["pmid"]):
                    title = title.strip()
                    abstract = abstract.strip()
                    # Don't index duplicate documents
                    if pmid in yielded:
                        continue
                    yielded.add(pmid)
                    yield {
                        "docno": f'{split}_{example["review_id"]}_{pmid}',
                        "text": f"{title.strip()} {abstract.strip()}",
                    }

    def get_topics(self, split: str, max_examples: Optional[int] = None) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        if max_examples:
            dataset = dataset[:max_examples]
        queries = dataset["background"]
        qids = dataset["review_id"]
        topics = pd.DataFrame({"qid": qids, "query": queries})
        return _sanitize_query(topics)

    def get_qrels(self, split: str) -> pd.DataFrame:
        dataset = self._hf_dataset[split]
        qids, docnos = [], []
        for example in dataset:
            qids.extend([example["review_id"]] * len(example["pmid"]))
            docnos.extend([f'{split}_{example["review_id"]}_{pmid}' for pmid in example["pmid"]])
        labels = [1] * len(qids)
        return pd.DataFrame({"qid": qids, "docno": docnos, "label": labels})

    def get_document_stats(self) -> Dict[str, float]:
        num_docs = []
        for split in self._hf_dataset:
            for example in self._hf_dataset[split]:
                num_docs.append(len(example["pmid"]))
        return {"max": np.max(num_docs), "mean": np.mean(num_docs), "min": np.min(num_docs)}
