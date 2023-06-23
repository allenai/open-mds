import json
from pathlib import Path

import flatten_dict
import typer
from rich import print
from rich.status import Status

from open_mds import metrics


def main(
    input_fp: str = typer.Argument(
        ..., help="Path to the input file containing model-generated and reference summaries"
    ),
    predictions_key: str = typer.Option(..., help="Key in the input file for the model-generated summaries"),
    references_key: str = typer.Option(..., help="Key in the input file for the reference summaries"),
    batch_size: int = typer.Option(64, help="Batch size to use when computing BERTScore"),
) -> None:
    """Evaluate model-generated summaries against reference summaries using ROUGE and BERTScore."""
    results = json.loads(Path(input_fp).read_text().strip())
    predictions = results[predictions_key]
    references = results[references_key]

    if len(predictions) != len(references):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) and references ({len(references)}) do not match."
        )

    with Status("Computing ROUGE scores"):
        rouge_results = metrics.compute_rouge(predictions=predictions, references=references)
    print(f"[green]Done.[/green] ROUGE-Avg: {rouge_results['rouge_avg_fmeasure_mean']:.2f}")
    with Status("Computing BERTScore"):
        bertscore_results = metrics.compute_bertscore(
            predictions=predictions, references=references, batch_size=batch_size
        )
    print(f"[green]Done.[/green] BERTScore F1: {bertscore_results['f1_mean']:.2f}")

    results.update(
        **flatten_dict.flatten(rouge_results, reducer="underscore"),
        **flatten_dict.flatten({"bertscore": bertscore_results}, reducer="underscore"),
    )

    with Status("Writing results to disk"):
        Path(input_fp).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Results written to '{input_fp}'")


if __name__ == "__main__":
    typer.run(main)
