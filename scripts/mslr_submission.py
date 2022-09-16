from enum import Enum
from pathlib import Path

import typer
from datasets import load_dataset


class Subtask(str, Enum):
    ms2 = "ms2"
    cochrane = "cochrane"


def main(
    generated_predictions_fp: str = typer.Argument(
        ..., help="Filepath to the generated_predictions.txt file produced by HuggingFace."
    ),
    submission_fp: str = typer.Argument(
        ..., help="Filepath to save the file for submission to the MSLR2022 leaderboard."
    ),
    subtask: Subtask = typer.Argument(..., case_sensitive=False, help="Which subtask to prepare the submission for."),
):
    """Format the generated_predictions.txt file produced by HuggingFace for submission to MSLR2022 leaderboard."""

    # Load the dataset in order to get the ReviewID's
    dataset = load_dataset("allenai/mslr2022", subtask, split="test")

    # Create a string containing the data in the format expected by the leaderboard
    submission = ",ReviewID,Generated\n"
    with open(generated_predictions_fp, "r") as f:
        for i, (example, line) in enumerate(zip(dataset, f)):
            submission += f'{i},{example["review_id"]},"{line.strip()}"\n'

    # Write the formatted data to a file, making sure its parent directory exists first
    Path(submission_fp).parent.mkdir(parents=True, exist_ok=True)
    with open(submission_fp, "w") as f:
        f.write(submission)


if __name__ == "__main__":
    typer.run(main)
