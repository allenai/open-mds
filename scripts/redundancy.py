import json
import statistics
from pathlib import Path
from random import sample
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import spacy
import typer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en.stop_words import STOP_WORDS
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

_PUBTATOR_API_URL = (
    "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator"
)
_WIKI_RANDOM_URL = "https://en.wikipedia.org/api/rest_v1/page/random/summary"

app = typer.Typer()


def _sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


def _rawcount(filepath: Path) -> int:
    """Returns the number of lines in the file at `filepath`.
    Adapted from: https://stackoverflow.com/a/27518377/6578628
    """
    f = open(filepath, "rb")
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b"\n")
        buf = read_f(buf_size)

    return lines


def _compute_fleiss_kappa(nlp: spacy.lang, texts: List[str]) -> Tuple[Optional[float], List[str]]:
    # Cleanup whitespace
    texts = (_sanitize_text(text) for text in texts)
    # Lemmatize and remove stopwords
    docs = nlp.pipe(texts)
    preproc_docs = [
        " ".join(token.lemma_ for token in doc if token.text.lower() not in STOP_WORDS)
        for doc in docs
    ]

    # Can't compute similarity if there are less than 2 documents
    if len(preproc_docs) < 2:
        return None, preproc_docs

    # Compute the Fleiss' kappa score for this instance (conidering up to 3-grams)
    vectorizer = CountVectorizer(strip_accents="unicode", ngram_range=(1, 3))
    X = vectorizer.fit_transform(preproc_docs)
    data, _ = aggregate_raters(X.toarray().T)
    fk = fleiss_kappa(data)

    return fk, preproc_docs


def _compute_cosine_similarity(model, texts: List[str]) -> float:
    """Computes the cosine similarity between the texts and the model."""
    texts = [_sanitize_text(text, lowercase=True) for text in texts]
    # Can't compute similarity if there are less than 2 documents
    if len(texts) < 2:
        return None
    paraphrases = paraphrase_mining(model, texts)
    sim = statistics.mean([paraphrase[0] for paraphrase in paraphrases])
    return sim


@app.command()
def random_words(
    output_filepath: Path = typer.Argument(
        ..., help="Path to save the output json lines file containing Fleiss' Kappa scores."
    ),
) -> None:
    """"""
    results = []

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    model = SentenceTransformer("all-distilroberta-v1")

    with typer.progressbar(range(2, 11), label="Computing Fleiss' Kappa") as pbar:
        for n in pbar:
            for _ in range(20):
                texts = [requests.get(_WIKI_RANDOM_URL).json()["extract"] for _ in range(n)]
                sim, preproc_docs = _compute_fleiss_kappa(nlp, texts)
                # sim = _compute_cosine_similarity(model, texts)

                # Aggregate results based on # of docs
                results.append({"num_docs": len(texts), "similarity": sim})

    df = pd.DataFrame(results)
    _ = sns.boxplot(x="num_docs", y="similarity", data=df)
    typer.secho("Displaying plot... exit to continue.", fg="blue")
    plt.show()

    # Write the results to a jsonl file.
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    output_filepath.write_text("\n".join(json.dumps(result) for result in results))


@app.command()
def multi_news(
    input_filepath: Path = typer.Argument(
        ..., help="Path to a file containing MS2 data e.g. path/to/val.txt.src."
    ),
    output_filepath: Path = typer.Argument(
        ..., help="Path to save the output json lines file containing Fleiss' Kappa scores."
    ),
    max_instances: int = typer.Option(
        None,
        help=(
            "Maximum number of instances to consider when producing the plot. Defaults to all"
            " instances in `filepath`."
        ),
    ),
) -> None:
    """Plots the Fleiss' Kappa for the included studies of each instance in MS2."""
    results = []

    # Get the number of instances in the dataset for a more informative progress bar
    num_instances = _rawcount(input_filepath)

    # Disable the components we don't need for lemmatization
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    model = SentenceTransformer("all-distilroberta-v1")

    with open(input_filepath) as f:
        # Get the text of the included studies by querying PubTator
        length = max_instances if max_instances is not None else num_instances
        with typer.progressbar(f, length=length, label="Computing Fleiss' Kappa") as pbar:
            for line in pbar:
                texts = line.split(" story_separator_special_tag ")

                # sim, preproc_docs = _compute_fleiss_kappa(nlp, texts)
                sim = _compute_cosine_similarity(model, texts)

                if sim is None:
                    continue

                # Aggregate results based on # of docs
                results.append(
                    {
                        # "docs": texts,
                        # "preproc_docs": preproc_docs,
                        "num_docs": len(texts),
                        "similarity": sim,
                    }
                )

                if max_instances and len(results) >= max_instances:
                    break

    df = pd.DataFrame(results)
    _ = sns.boxplot(x="num_docs", y="similarity", data=df)
    typer.secho("Displaying plot... exit to continue.", fg="blue")
    plt.show()

    # Write the results to a jsonl file.
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    output_filepath.write_text("\n".join(json.dumps(result) for result in results))


@app.command()
def ms2(
    input_filepath: Path = typer.Argument(
        ...,
        help="Path to a file containing MS2 data (e.g. path/to/validation_reviews.jsonl",
    ),
    output_filepath: Path = typer.Argument(
        ..., help="Path to save the output json lines file containing Fleiss' Kappa scores."
    ),
    max_instances: int = typer.Option(
        None,
        help=(
            "Maximum number of instances to consider when producing the plot. Defaults to all"
            " instances in `filepath`."
        ),
    ),
) -> None:
    """Plots the Fleiss' Kappa for the included studies of each instance in MS2."""
    results = []

    # Get the number of instances in the dataset for a more informative progress bar
    num_instances = _rawcount(input_filepath)

    # Disable the components we don't need for lemmatization
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    model = SentenceTransformer("all-distilroberta-v1")

    with open(input_filepath) as f:
        # Get the text of the included studies by querying PubTator
        length = max_instances if max_instances is not None else num_instances
        with typer.progressbar(f, length=length, label="Computing Fleiss' Kappa") as pbar:
            for line in pbar:
                included_studies = json.loads(line)["included_studies"]
                pmids = [study["pmid"] for study in included_studies]
                body = {"type": "pmid", "pmids": pmids, "concepts": "none"}
                r = requests.post(_PUBTATOR_API_URL, json=body)

                # Format this text as title + SPACE + abstract
                texts = []
                for article in r.text.strip().split("\n\n"):
                    if not article:
                        continue
                    title, abstract = article.split("\n")
                    title = title.split("|t|")[1]
                    abstract = abstract.split("|a|")[1]
                    texts.append(f"{title.strip()} {abstract.strip()}")

                # sim, preproc_docs = _compute_fleiss_kappa(nlp, texts)
                sim = _compute_cosine_similarity(model, texts)

                if sim is None:
                    continue

                # Aggregate results based on # of docs
                results.append(
                    {
                        # "docs": texts,
                        # "preproc_docs": preproc_docs,
                        "num_docs": len(texts),
                        "similarity": sim,
                    }
                )

                if max_instances and len(results) >= max_instances:
                    break

    df = pd.DataFrame(results)
    _ = sns.boxplot(x="num_docs", y="similarity", data=df)
    typer.secho("Displaying plot... exit to continue.", fg="blue")
    plt.show()

    # Write the results to a jsonl file.
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)
    output_filepath.write_text("\n".join(json.dumps(result) for result in results))


@app.command()
def plot(input_dir: Path) -> None:
    results = []
    for filepath in Path(input_dir).glob("*.jsonl"):
        for line in filepath.read_text().strip().splitlines():
            result = json.loads(line)
            result["dataset"] = filepath.stem
            if result["num_docs"] <= 10:
                results.append(result)
    df = pd.DataFrame(results)
    _ = sns.boxplot(x="num_docs", y="similarity", hue="dataset", data=df).set(
        title=Path(input_dir).name
    )
    typer.secho("Displaying plot... exit to continue.", fg="blue")
    plt.show()


if __name__ == "__main__":
    app()
