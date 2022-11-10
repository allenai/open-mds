from retrieval_exploration import indexing
from retrieval_exploration.common import util
import pandas as pd


def test_sanitize_query() -> None:
    topics = pd.DataFrame({"qid": [1], "query": [r"This is a query with special characters /\%^&*()"]})
    actual = indexing._sanitize_query(topics)
    # PyTerrier will retain the old query
    assert actual["query_0"].iloc[0] == topics["query"].iloc[0]
    # Query always contains special characters, so it won't match the old query
    assert actual["query"].iloc[0] != topics["query"].iloc[0]


class TestCanonicalMDSDataset:
    def test_info_url(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset):
        assert (
            canonical_mds_pt_dataset.info_url() == f"https://huggingface.co/datasets/{canonical_mds_pt_dataset.path}"
        )

    def test_replace(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        # Choose a query
        split = "train"
        idx = 0
        qid = f"{split}_{idx}"

        # Create dummy retrieval results
        retrieved = pd.DataFrame({"qid": [qid], "docno": ["validation_0_0"]})
        expected_document = canonical_mds_pt_dataset._hf_dataset["validation"][0]["document"].split(
            canonical_mds_pt_dataset._doc_sep_token
        )[0]

        # Check that the example is modified as expected
        example = {"document": "This can be anything", "summary": "This could be anything"}
        actual = canonical_mds_pt_dataset.replace(example, idx, split=split, retrieved=retrieved)
        assert actual["document"].strip() == expected_document.strip()
        # Check that the target summary is not modified
        assert actual["summary"] == example["summary"]
        # Check that the modified example contains only expected keys
        assert all(key in ["document", "summary"] for key in actual)

    def test_get_corpus_iter(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_docno = "train_0_0"
        expected_text = util.split_docs(
            canonical_mds_pt_dataset._hf_dataset["train"]["document"][0],
            doc_sep_token=canonical_mds_pt_dataset._doc_sep_token,
        )[0]
        corpus_iter = canonical_mds_pt_dataset.get_corpus_iter(verbose=False)
        # Check that the first yielded document is as expected
        first_item = next(corpus_iter)
        assert first_item == {"docno": expected_docno, "text": expected_text}

    def test_get_topics(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_qid = "train_0"
        expected_query = canonical_mds_pt_dataset._hf_dataset["train"]["summary"][0]
        topics = canonical_mds_pt_dataset.get_topics(split="train", max_examples=1)
        # Check that the first query is as expected
        assert topics["qid"].tolist() == [expected_qid]
        # Because the query is sanitized, check they original query
        assert topics["query_0"].tolist() == [expected_query]

    def test_get_qrels(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        qrels = canonical_mds_pt_dataset.get_qrels(split="validation")
        expected_qid = "validation_0"
        expected_docnos = "validation_0_0"
        expected_labels = 1
        # Check that the first labelled example is as expected
        assert qrels["qid"].iloc[0] == expected_qid
        assert qrels["docno"].iloc[0] == expected_docnos
        assert qrels["label"].iloc[0] == expected_labels

    def test_get_document_stats(self, canonical_mds_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        document_stats = canonical_mds_pt_dataset.get_document_stats(avg_tokens_per_doc=True)

        if canonical_mds_pt_dataset.path == "multi_news":
            assert document_stats["max"] == 10
            assert round(document_stats["mean"], 2) == 2.75
            assert document_stats["min"] == 1
            assert round(document_stats["avg_tokens_per_doc"]) == 788
        elif canonical_mds_pt_dataset.path == "ccdv/WCEP-10":
            assert document_stats["max"] == 10
            assert round(document_stats["mean"], 2) == 9.07
            assert document_stats["min"] == 1
            assert round(document_stats["avg_tokens_per_doc"]) == 494


class TestMultiXScienceDataset:
    def test_info_url(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset):
        assert multxscience_pt_dataset.info_url() == "https://huggingface.co/datasets/multi_x_science_sum"

    def test_replace(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        # Choose a query
        split = "train"
        idx = 0
        qid = f"{split}_{idx}"

        # Create dummy retrieval results
        expected_docno = multxscience_pt_dataset._hf_dataset["validation"][0]["ref_abstract"]["mid"][0]
        expected_document = multxscience_pt_dataset._hf_dataset["validation"][0]["ref_abstract"]["abstract"][0]
        retrieved = pd.DataFrame({"qid": [qid], "docno": [expected_docno]})

        # Check that the example is modified as expected
        example = {
            "abstract": "This can be anything",
            "related_work": "This can be anything",
            "ref_abstract": {"mid": [expected_docno], "abstract": ["This can be anything"]},
        }
        actual = multxscience_pt_dataset.replace(example, idx, split=split, retrieved=retrieved)
        assert actual["ref_abstract"]["mid"] == [expected_docno]
        assert actual["ref_abstract"]["abstract"] == [expected_document]
        # Check that the target summary is not modified
        assert actual["related_work"] == example["related_work"]
        # Check that the modified example contains only expected keys
        assert all(key in ["abstract", "related_work", "ref_abstract"] for key in actual)

    def test_get_corpus_iter(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_docno = multxscience_pt_dataset._hf_dataset["train"][0]["ref_abstract"]["mid"][0]
        expected_text = multxscience_pt_dataset._hf_dataset["train"][0]["ref_abstract"]["abstract"][0]
        corpus_iter = multxscience_pt_dataset.get_corpus_iter(verbose=False)
        # Check that the first yielded document is as expected
        first_item = next(corpus_iter)
        assert first_item == {"docno": expected_docno, "text": expected_text}

    def test_get_topics(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_qid = "train_0"
        expected_query = multxscience_pt_dataset._hf_dataset["train"]["related_work"][0]
        topics = multxscience_pt_dataset.get_topics(split="train", max_examples=1)
        # Check that the first query is as expected
        assert topics["qid"].tolist() == [expected_qid]
        # Because the query is sanitized, check they original query
        assert topics["query_0"].tolist() == [expected_query]

    def test_get_qrels(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        qrels = multxscience_pt_dataset.get_qrels(split="validation")
        expected_qid = "validation_0"
        expected_docnos = multxscience_pt_dataset._hf_dataset["validation"][0]["ref_abstract"]["mid"][0]
        expected_labels = 1
        # Check that the first labelled example is as expected
        assert qrels["qid"].iloc[0] == expected_qid
        assert qrels["docno"].iloc[0] == expected_docnos
        assert qrels["label"].iloc[0] == expected_labels

    def test_get_document_stats(self, multxscience_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        document_stats = multxscience_pt_dataset.get_document_stats(avg_tokens_per_doc=True)
        assert document_stats["max"] == 20
        assert round(document_stats["mean"], 2) == 4.08
        assert document_stats["min"] == 1
        assert round(document_stats["avg_tokens_per_doc"]) == 153


class TestMSLR2022MS2Dataset:
    def test_info_url(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        assert ms2_pt_dataset.info_url() == "https://huggingface.co/datasets/allenai/mslr2022"

    def test_replace(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        # Choose a query
        split = "train"
        idx = 0
        qid = ms2_pt_dataset._hf_dataset["train"]["review_id"][idx]

        # Create dummy retrieval results
        expected_docno = ms2_pt_dataset._hf_dataset["validation"][0]["pmid"][0]
        expected_title = ms2_pt_dataset._hf_dataset["validation"][0]["title"][0]
        expected_abstract = ms2_pt_dataset._hf_dataset["validation"][0]["abstract"][0]
        retrieved = pd.DataFrame({"qid": [qid], "docno": [expected_docno]})

        # Check that the example is modified as expected
        example = {
            "review_id": qid,
            "background": "This can be anything",
            "pmid": [expected_docno],
            "title": [expected_title],
            "abstract": [expected_abstract],
        }
        actual = ms2_pt_dataset.replace(example, idx, split=split, retrieved=retrieved)
        assert actual["pmid"] == [expected_docno]
        assert actual["title"] == [expected_title]
        assert actual["abstract"] == [expected_abstract]
        # Check that the other fields are not modified
        assert actual["review_id"] == example["review_id"]
        assert actual["background"] == example["background"]
        # Check that the modified example contains only expected keys
        assert all(key in ["review_id", "background", "pmid", "title", "abstract"] for key in actual)

    def test_get_corpus_iter(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset):
        expected_docno = ms2_pt_dataset._hf_dataset["train"][0]["pmid"][0]
        title = ms2_pt_dataset._hf_dataset["train"][0]["title"][0]
        abstract = ms2_pt_dataset._hf_dataset["train"][0]["abstract"][0]
        expected_text = f"{title} {abstract}"
        corpus_iter = ms2_pt_dataset.get_corpus_iter(verbose=False)
        # Check that the first yielded document is as expected
        first_item = next(corpus_iter)
        assert first_item == {"docno": expected_docno, "text": expected_text}

    def test_get_topics(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_qid = ms2_pt_dataset._hf_dataset["train"]["review_id"][0]
        expected_query = ms2_pt_dataset._hf_dataset["train"]["background"][0]
        topics = ms2_pt_dataset.get_topics(split="train", max_examples=1)
        # Check that the first query is as expected
        assert topics["qid"].tolist() == [expected_qid]
        # Because the query is sanitized, check they original query
        assert topics["query_0"].tolist() == [expected_query]

    def test_get_qrels(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        qrels = ms2_pt_dataset.get_qrels(split="validation")
        expected_qid = ms2_pt_dataset._hf_dataset["validation"]["review_id"][0]
        expected_docnos = ms2_pt_dataset._hf_dataset["validation"][0]["pmid"][0]
        expected_labels = 1
        # Check that the first labelled example is as expected
        assert qrels["qid"].iloc[0] == expected_qid
        assert qrels["docno"].iloc[0] == expected_docnos
        assert qrels["label"].iloc[0] == expected_labels

    def test_get_document_stats(self, ms2_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        document_stats = ms2_pt_dataset.get_document_stats()
        assert document_stats["max"] == 401
        assert round(document_stats["mean"], 2) == 23.23
        assert document_stats["min"] == 1

        # Check that we can cap the max number of documents
        document_stats = ms2_pt_dataset.get_document_stats(max_documents=25, avg_tokens_per_doc=True)
        assert document_stats["max"] == 25
        assert round(document_stats["mean"], 2) == 16.57
        assert document_stats["min"] == 1
        assert round(document_stats["avg_tokens_per_doc"]) == 332


class TestMSLR2022CochraneDataset:
    def test_info_url(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        assert cochrane_pt_dataset.info_url() == "https://huggingface.co/datasets/allenai/mslr2022"

    def test_replace(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        # Choose a query
        split = "train"
        idx = 0
        qid = cochrane_pt_dataset._hf_dataset["train"]["review_id"][idx]

        # Create dummy retrieval results
        expected_docno = cochrane_pt_dataset._hf_dataset["validation"][0]["pmid"][0]
        expected_title = cochrane_pt_dataset._hf_dataset["validation"][0]["title"][0]
        expected_abstract = cochrane_pt_dataset._hf_dataset["validation"][0]["abstract"][0]
        retrieved = pd.DataFrame({"qid": [qid], "docno": [expected_docno]})

        # Check that the example is modified as expected
        example = {
            "review_id": qid,
            "pmid": [expected_docno],
            "title": [expected_title],
            "abstract": [expected_abstract],
        }
        actual = cochrane_pt_dataset.replace(example, idx, split=split, retrieved=retrieved)
        assert actual["pmid"] == [expected_docno]
        assert actual["title"] == [expected_title]
        assert actual["abstract"] == [expected_abstract]
        # Check that the other fields are not modified
        assert actual["review_id"] == example["review_id"]
        # Check that the modified example contains only expected keys
        assert all(key in ["review_id", "pmid", "title", "abstract"] for key in actual)

    def test_get_corpus_iter(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset):
        expected_docno = cochrane_pt_dataset._hf_dataset["train"][0]["pmid"][0]
        title = cochrane_pt_dataset._hf_dataset["train"][0]["title"][0]
        abstract = cochrane_pt_dataset._hf_dataset["train"][0]["abstract"][0]
        expected_text = f"{title} {abstract}"
        corpus_iter = cochrane_pt_dataset.get_corpus_iter(verbose=False)
        # Check that the first yielded document is as expected
        first_item = next(corpus_iter)
        assert first_item == {"docno": expected_docno, "text": expected_text}

    def test_get_topics(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        expected_qid = cochrane_pt_dataset._hf_dataset["train"]["review_id"][0]
        expected_query = cochrane_pt_dataset._hf_dataset["train"]["target"][0]
        topics = cochrane_pt_dataset.get_topics(split="train", max_examples=1)
        # Check that the first query is as expected
        assert topics["qid"].tolist() == [expected_qid]
        # Because the query is sanitized, check they original query
        assert topics["query_0"].tolist() == [expected_query]

    def test_get_qrels(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        qrels = cochrane_pt_dataset.get_qrels(split="validation")
        expected_qid = cochrane_pt_dataset._hf_dataset["validation"]["review_id"][0]
        expected_docnos = cochrane_pt_dataset._hf_dataset["validation"][0]["pmid"][0]
        expected_labels = 1
        # Check that the first labelled example is as expected
        assert qrels["qid"].iloc[0] == expected_qid
        assert qrels["docno"].iloc[0] == expected_docnos
        assert qrels["label"].iloc[0] == expected_labels

    def test_get_document_stats(self, cochrane_pt_dataset: indexing.HuggingFacePyTerrierDataset) -> None:
        document_stats = cochrane_pt_dataset.get_document_stats()
        assert document_stats["max"] == 537
        assert round(document_stats["mean"], 2) == 10.91
        assert document_stats["min"] == 1

        # Check that we can cap the max number of documents
        document_stats = cochrane_pt_dataset.get_document_stats(max_documents=25, avg_tokens_per_doc=True)
        assert document_stats["max"] == 25
        assert round(document_stats["mean"], 2) == 8.9
        assert document_stats["min"] == 1
        assert round(document_stats["avg_tokens_per_doc"]) == 266
