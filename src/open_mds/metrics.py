import warnings
from typing import Any, Dict, List, Tuple

import nltk
import numpy as np
from datasets import load_metric

# The underlying language model used by BERTScore to compute the score
BERTSCORE_MODEL_TYPE = "microsoft/deberta-xlarge-mnli"


def _postprocess_text(*, predictions: List[str], references: List[str]) -> Tuple[List[str], List[str]]:
    """Simple post-processing of the text to make it compatible with the summarization evaluation metrics."""

    # Clean text by removing whitespace, newlines and tabs
    predictions = [" ".join(pred.strip().split()) for pred in predictions]
    references = [" ".join(ref.strip().split()) for ref in references]

    # rougeLSum expects newline after each sentence
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    references = ["\n".join(nltk.sent_tokenize(ref)) for ref in references]

    return predictions, references


def compute_rouge(*, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
    """Computes ROUGE scores using the datasets package."""
    rouge = load_metric("rouge")

    predictions, references = _postprocess_text(predictions=predictions, references=references)

    # Compute and post-process rouge results
    results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True,
        use_aggregator=False,
        **kwargs,
    )
    for key, value in results.items():
        results[key] = {
            "precision": [score.precision * 100 for score in value],
            "recall": [score.recall * 100 for score in value],
            "fmeasure": [score.fmeasure * 100 for score in value],
            "fmeasure_mean": np.mean([score.fmeasure for score in value]) * 100,
        }
    # Compute the arithmetic mean of ROUGE-1, ROUGE-2 and ROUGE-L following: https://arxiv.org/abs/2110.08499
    if all(rouge_type in results for rouge_type in ["rouge1", "rouge2", "rougeL"]):
        results["rouge_avg_fmeasure"] = np.mean(
            [results[key]["fmeasure"] for key in ["rouge1", "rouge2", "rougeL"]], axis=0
        ).tolist()
        results["rouge_avg_fmeasure_mean"] = np.mean(results["rouge_avg_fmeasure"]).item()
    else:
        warnings.warn(
            "ROUGE-1, ROUGE-2 and ROUGE-L are not all present in the results. Skipping the computation of ROUGE-AVG."
        )

    return results


def compute_bertscore(*, predictions: List[str], references: List[str], **kwargs):
    """Computes BERTScore using the datasets package."""
    bertscore = load_metric("bertscore")

    predictions, references = _postprocess_text(predictions=predictions, references=references)

    # Compute and post-process bertscore results
    results = bertscore.compute(
        predictions=predictions,
        references=references,
        # These are mostly based on the recommendations in https://github.com/Tiiiger/bert_score
        model_type=BERTSCORE_MODEL_TYPE,
        lang="en",
        rescale_with_baseline=True,
        use_fast_tokenizer=True,
        **kwargs,
    )
    results["f1_mean"] = np.mean(results["f1"])
    for key, value in results.items():
        if key == "hashcode":
            continue
        if isinstance(value, list):
            results[key] = [score * 100 for score in value]
        else:
            results[key] = value * 100

    return results
