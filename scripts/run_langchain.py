import json
import os
from pathlib import Path

import tiktoken
import typer
from datasets import load_dataset
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track

from open_mds import metrics
from open_mds.common import util

DOC_SEP_TOKEN = "<|doc|>"


def _print_example_prompt(llm, example_prompt, example_printed: bool) -> bool:
    """Print the example prompt if it hasn't already been printed."""
    if not example_printed:
        print(f"Example prompt (length={llm.get_num_tokens(example_prompt)}): {example_prompt}")
    return True


def main(
    dataset_name: str = typer.Argument("The name of the dataset to use (via the datasets library)."),
    output_fp: str = typer.Argument("Filepath to save the results to."),
    dataset_config_name: str = typer.Option(
        None, help="The configuration name of the dataset to use (via the datasets library)."
    ),
    openai_api_key: str = typer.Option(
        None, help="OpenAI API key. If None, we assume this is set via the OPENAI_API_KEY enviornment variable."
    ),
    model_name: str = typer.Option(
        "gpt-3.5-turbo", help="A valid OpenAI API model. See: https://platform.openai.com/docs/models"
    ),
    temperature: float = typer.Option(
        0.7,
        help="The temperature to use when sampling from the model. See: https://platform.openai.com/docs/api-reference/completions",
    ),
    max_input_tokens: int = typer.Option(
        3073,
        help="The maximum number of tokens to allow in the models input prompt.",
    ),
    max_output_Tokens: int = typer.Option(
        1024,
        help="The maximum number of tokens to generate in the chat completion. See: https://platform.openai.com/docs/api-reference/completions",
    ),
    max_examples: int = typer.Option(
        None,
        help="The maximum number of examples to use from the dataset. Helpful for debugging before commiting to a full run.",
    ),
    split: str = typer.Option("test", help="The dataset split to use."),
):
    """Evaluate an OpenAI based large language model for multi-document summarization."""

    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config_name, split=split)
    print(f'Loaded dataset "{dataset_name}" (config="{dataset_config_name}", split="{split}")')

    # Setup the LLM
    openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "OpenAI API key must be provided via the OPENAI_API_KEY environment variable or the --openai-api-key flag."
        )
    llm = ChatOpenAI(
        model=model_name, temperature=temperature, openai_api_key=openai_api_key, max_tokens=max_output_Tokens
    )
    tokenizer = tiktoken.encoding_for_model(model_name)
    print(f'Using "{model_name}" as the LLM with temperature={temperature} and {max_output_Tokens} max output tokens.')

    # Setup the prompt
    if dataset_name == "multi_news" or "multinews" in dataset_name:
        prompt = PromptTemplate(
            input_variables=["documents"],
            template="""Given multiple news articles written about a particular event, write a short summary of approximately 270 words or less.

Example summary 1: – The unemployment rate dropped to 8.2% last month, but the economy only added 120,000 jobs, when 203,000 new jobs had been predicted, according to today's jobs report. Reaction on the Wall Street Journal's MarketBeat Blog was swift: "Woah!!! Bad number." The unemployment rate, however, is better news; it had been expected to hold steady at 8.3%. But the AP notes that the dip is mostly due to more Americans giving up on seeking employment.
Example summary 2: – A twin-engine Embraer jet that the FAA describes as "on approach to Runway 14" at the Montgomery County Airpark in Gaithersburg, Maryland, crashed into a home this morning, engulfing that home in flames and setting two others on fire. Three people are dead, but the count could grow. A Montgomery County Fire rep says three fliers were killed in the crash, but notes the corporate plane may have had a fourth person on board, reports the AP. A relative of the owner of the home that was hit tells WUSA 9 that a mother with three children pre-school age and under should have been home at the time; there's no word on the family's whereabouts. The crash occurred around 11am on Drop Forge Lane, and the fire was extinguished within an hour. Crews are now searching the wreckage. A witness noted the plane appeared to "wobble" before the crash; the airport is no more than 3/4 mile from the crash scene. NTSB and FAA will investigate.

{documents}\nSummary:",
        """,
        )
    elif dataset_name == "multi_x_science_sum" or "multixscience" in dataset_name:
        prompt = PromptTemplate(
            input_variables=["abstract", "ref_abstract"],
            template="""Given the abstract of a scientific paper and the abstracts of some papers it cites, write a short related works section of approximately 100 words or less.
Example:

Abstract: We give a purely topological definition of the perturbative quantum invariants of links and 3-manifolds associated with Chern-Simons field theory. Our definition is as close as possible to one given by Kontsevich. We will also establish some basic properties of these invariants, in particular that they are universally finite type with respect to algebraically split surgery and with respect to Torelli surgery. Torelli surgery is a mutual generalization of blink surgery of Garoufalidis and Levine and clasper surgery of Habiro.
Referenced abstract @cite_16: This note is a sequel to our earlier paper of the same title [4] and describes invariants of rational homology 3-spheres associated to acyclic orthogonal local systems. Our work is in the spirit of the Axelrod–Singer papers [1], generalizes some of their results, and furnishes a new setting for the purely topological implications of their work.
Referenced abstract @cite_26: Recently, Mullins calculated the Casson-Walker invariant of the 2-fold cyclic branched cover of an oriented link in S^3 in terms of its Jones polynomial and its signature, under the assumption that the 2-fold branched cover is a rational homology 3-sphere. Using elementary principles, we provide a similar calculation for the general case. In addition, we calculate the LMO invariant of the p-fold branched cover of twisted knots in S^3 in terms of the Kontsevich integral of the knot.
Related work: Two other generalizations that can be considered are invariants of graphs in 3-manifolds, and invariants associated to other flat connections @cite_16 . We will analyze these in future work. Among other things, there should be a general relation between flat bundles and links in 3-manifolds on the one hand and finite covers and branched covers on the other hand @cite_26.

Abstract: {abstract}\n{ref_abstract}\nRelated work:",
        """,
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name} or config {dataset_config_name}")

    # Setup the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    outputs = []
    example_printed = False
    for example in track(dataset, description="Generating summaries", total=max_examples or len(dataset)):
        # Format the inputs, truncate, and sanitize
        if dataset_name == "multi_news" or "multinews" in dataset_name:
            documents, summary = util.preprocess_multi_news(
                example["document"], example["summary"], doc_sep_token=DOC_SEP_TOKEN
            )
            documents, summary = util.sanitize_text(documents), util.sanitize_text(summary)
            documents = util.truncate_multi_doc(
                documents,
                doc_sep_token=DOC_SEP_TOKEN,
                max_length=max_input_tokens - llm.get_num_tokens(prompt.format(documents="")),
                tokenizer=tokenizer,
            )
            documents = "\n".join(
                f"Source {i+1}: {doc}" for i, doc in enumerate(util.split_docs(documents, doc_sep_token=DOC_SEP_TOKEN))
            )
            # Print the first example, helpful for debugging / catching errors in the prompt
            example_prompt = prompt.format(documents=documents)
            example_printed = _print_example_prompt(llm, example_prompt, example_printed)
            # Run the chain
            output = chain.run(documents=documents)
        else:
            abstract = util.sanitize_text(example["abstract"])
            ref_abstract = DOC_SEP_TOKEN.join(
                f"Referenced abstract {cite_n}: {util.sanitize_text(ref_abs)}"
                for cite_n, ref_abs in zip(example["ref_abstract"]["cite_N"], example["ref_abstract"]["abstract"])
                if ref_abs.strip()
            )
            ref_abstract = util.truncate_multi_doc(
                ref_abstract,
                doc_sep_token=DOC_SEP_TOKEN,
                max_length=max_input_tokens - llm.get_num_tokens(prompt.format(abstract=abstract, ref_abstract="")),
                tokenizer=tokenizer,
            )
            ref_abstract = ref_abstract.replace(DOC_SEP_TOKEN, "\n")
            example_prompt = prompt.format(abstract=abstract, ref_abstract=ref_abstract)
            example_printed = _print_example_prompt(llm, example_prompt, example_printed)
            output = chain.run(abstract=abstract, ref_abstract=ref_abstract)

        outputs.append(output)
        if max_examples and len(outputs) >= max_examples:
            break

    # Compute the metrics and save the results
    if dataset_name == "multi_news" or "multinews" in dataset_name:
        references = dataset["summary"]
    else:
        references = dataset["related_work"]

    rouge = metrics.compute_rouge(predictions=outputs, references=references[: len(outputs)])
    bertscore = metrics.compute_bertscore(predictions=outputs, references=references[: len(outputs)])
    results = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "model_name": model_name,
        "temperature": temperature,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_Tokens,
        "max_examples": max_examples,
        "outputs": outputs,
        "rogue": rouge,
        "bertscore": bertscore,
    }
    Path(output_fp).parent.mkdir(exist_ok=True, parents=True)
    Path(output_fp).write_text(json.dumps(results, indent=2))
    print(f"Results written to {output_fp}")


if __name__ == "__main__":
    typer.run(main)
