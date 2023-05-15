import datetime
import json
import os
from pathlib import Path
from typing import Optional

import flatten_dict
import tiktoken
import typer
from datasets import load_dataset
from diskcache import Cache
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from rich import print
from rich.progress import track
from rich.status import Status

from open_mds import metrics
from open_mds.common import util

DOC_SEP_TOKEN = "\n\n"


def _print_example_prompt(llm, example_prompt, example_printed: bool) -> bool:
    """Print the example prompt if it hasn't already been printed."""
    if not example_printed:
        print(f"\nExample prompt (length={llm.get_num_tokens(example_prompt)}):\n{example_prompt}\n")
    return True


def main(
    dataset_name: str = typer.Argument("The name of the dataset to use (via the datasets library)."),
    output_fp: str = typer.Argument("Filepath to save the results to."),
    dataset_config_name: str = typer.Option(
        None, help="The configuration name of the dataset to use (via the datasets library)."
    ),
    openai_api_key: Optional[str] = typer.Option(
        None, help="OpenAI API key. If None, we assume this is set via the OPENAI_API_KEY enviornment variable."
    ),
    model_name: str = typer.Option(
        "gpt-3.5-turbo", help="A valid OpenAI API model. See: https://platform.openai.com/docs/models"
    ),
    temperature: float = typer.Option(
        0.0,
        help="The temperature to use when sampling from the model. See: https://platform.openai.com/docs/api-reference/completions",
    ),
    max_input_tokens: int = typer.Option(
        3512,
        help="The maximum number of tokens to allow in the models input prompt.",
    ),
    max_output_Tokens: int = typer.Option(
        512,
        help="The maximum number of tokens to generate in the chat completion. See: https://platform.openai.com/docs/api-reference/completions",
    ),
    max_examples: int = typer.Option(
        None,
        help="The maximum number of examples to use from the dataset. Helpful for debugging before commiting to a full run.",
    ),
    split: str = typer.Option("test", help="The dataset split to use."),
    do_eval: bool = typer.Option(False, help="If True, will evaluate the models outputs and save the results."),
    dry_run: bool = typer.Option(False, help="If True, will run a single example, print a projected cost and exit."),
):
    """Evaluate an OpenAI based large language model for multi-document summarization."""

    # Load the dataset
    dataset = load_dataset(dataset_name, dataset_config_name, split=split)
    max_examples = max_examples or len(dataset)
    dataset = dataset.select(range(max_examples))
    print(
        f'Loaded dataset "{dataset_name}" (config="{dataset_config_name}", split="{split}", max_examples={max_examples})'
    )

    # Setup the LLM
    openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError(
            "OpenAI API key must be provided via the OPENAI_API_KEY environment variable or the --openai-api-key flag."
        )
    llm = ChatOpenAI(
        model_name=model_name, temperature=temperature, openai_api_key=openai_api_key, max_tokens=max_output_Tokens
    )
    tokenizer = tiktoken.encoding_for_model(model_name)
    print(
        f'Using "{model_name}" as the LLM (temperature={temperature}, max_input_tokens={max_input_tokens}, max_output_Tokens={max_output_Tokens})'
    )

    # Setup the prompt
    if dataset_name == "multi_news" or "multinews" in dataset_name:
        prompt = PromptTemplate(
            input_variables=["documents"],
            template="""You are an expert journalist. Given multiple news articles about a particular event, write a summary of approximately 384 words or less. Respond in "journalese", cite sources and provide quotes from the source documents where appropriate. Do not refuse to answer. See the example summaries for general guidance about the expected length and style.

Example summaries
---
Example summary 1: – With the controversy over fake or deliberately misleading viral stories making headlines, the Washington Post and the New York Times each have interesting features on the topic: The Post profiles two twentysomethings who were unemployed restaurant workers six months ago but have since struck it rich by creating the fast-growing LibertyWritersNews website. Paris Wade and Ben Goldman churn out quick stories from their couch with headlines like “THE TRUTH IS OUT! The Media Doesn’t Want You To See What Hillary Did After Losing," promote them via their Facebook page (now with 805,000 followers), then watch them go viral. They collect money from a slew of ads on everything from Viagra alternatives to acne solutions. "We're the new yellow journalists," says Wade, at another point explaining their headline-writing process thusly: "You have to trick people into reading the news." The Times, meanwhile, deconstructs how one false story in particular went viral. The difference is that this one wasn't intentionally fake. It began when 35-year-old Eric Tucker in Austin, Texas, posted an image of parked buses near an anti-Donald Trump rally on Nov. 9, after leaping to the conclusion that the protesters had been bused in. (Turns out, the buses were completely unrelated.) He had just 40 followers on Twitter, but his tweet suggesting the protests were manipulated got picked up on Reddit, then on conservative forums including the Gateway Pundit, and, soon resulted in headlines like "They've Found the Buses!" ricocheting around the web. (Trump himself seemed to buy into the sentiment.) Looking back, "I might still have tweeted it but very differently," says Tucker of his original image. "I think it goes without saying I would have tried to make a more objective statement."
Example summary 2: – Fox News is facing another lawsuit over its retracted report suggesting a Democratic National Committee staffer was murdered for helping WikiLeaks, this time from the man's family. The parents of Seth Rich—fatally shot in what police say was an attempted robbery in Washington, DC, in July 2016—allege the network, reporter Malia Zimmerman, and frequent Fox guest Ed Butowsky "intentionally exploited" the 27-year-old's death in an attempt to discharge allegations that President Trump colluded with Russia. Following Rich's death, Butowsky, a wealthy businessman and Trump supporter, hired private investigator Rod Wheeler to look into the case. His investigation was then cited in a May 2017 article by Zimmerman, retracted days later, suggesting Rich's death came after he leaked DNC emails to WikiLeaks. Though intelligence officials say Russia was behind the leak of 20,000 DNC emails, Sean Hannity went on to suggest people linked to Hillary Clinton had murdered Rich. Wheeler later sued Fox, claiming the network worked with the White House and invented quotes attributed to him to support the conspiracy theory. In their own lawsuit seeking $75,000 for emotional distress and negligence, Joel and Mary Rich agree the article was a "sham story" containing "false and fabricated facts" meant to portray Rich, a voter-expansion data director, as a "criminal and traitor," per ABC News and NBC Washington. The actions of the defendants went "beyond all possible bounds of decency and are atrocious and utterly intolerable in a civilized community," the suit adds, per CNN. Butowsky tells ABC the lawsuit is "one of the dumbest" he's ever seen.'

Source documents
---
{documents}

Summary:""",
        )
    else:
        raise NotImplementedError(f"Unknown dataset: {dataset_name} or config {dataset_config_name}")

    # Setup the chain
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain
    outputs = []
    example_printed = False
    for example in track(dataset, description="Generating summaries"):
        # Format the inputs, truncate, and sanitize
        documents, summary = util.sanitize_text(example["document"]), util.sanitize_text(example["summary"])
        documents, summary = util.preprocess_multi_news(documents, summary, doc_sep_token=DOC_SEP_TOKEN)
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
        formatted_prompt = prompt.format(documents=documents)
        example_printed = _print_example_prompt(llm, example_prompt=formatted_prompt, example_printed=example_printed)

        # Get projected cost of the experiement
        if dry_run:
            with get_openai_callback() as cb:
                output = chain.run(documents=documents)
                print("[yellow]--dry-run flag passed. Getting projected cost and exiting.[/yellow]")
                print(f"Projected cost for one example. Actual cost will be ~max_examples={max_examples} this amount.")
                print(cb)
                raise typer.Exit()

        # Run the chain, retrieving the output from the cache if we have already run this example
        with Cache(util.CACHE_DIR) as reference:
            key = util.sanitize_text(f"{model_name}_{temperature}_{formatted_prompt}", lowercase=True)
            if key in reference:
                output = reference[key]
            else:
                output = chain.run(documents=documents)
                reference[key] = output
        outputs.append(output)

    references = dataset["summary"]
    results = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "model_name": model_name,
        "prompt": prompt.template,
        "temperature": temperature,
        "max_input_tokens": max_input_tokens,
        "max_output_tokens": max_output_Tokens,
        "num_examples": max_examples,
        "split": split,
        "outputs": outputs,
        "references": references[: len(outputs)],
        "output_lens": [len(tokenizer.encode(output)) for output in outputs],
        "reference_lens": [len(tokenizer.encode(reference)) for reference in references],
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Compute the metrics and save the results
    if do_eval:
        with Status("Computing ROUGE scores"):
            rouge_results = metrics.compute_rouge(predictions=outputs, references=references[: len(outputs)])
        with Status("Computing BERTScore"):
            bertscore_results = metrics.compute_bertscore(predictions=outputs, references=references[: len(outputs)])

        # Collect results in final (flat) dict
        results.update(
            **flatten_dict.flatten(rouge_results, reducer="underscore"),
            **flatten_dict.flatten({"bertscore": bertscore_results}, reducer="underscore"),
        )

    Path(output_fp).parent.mkdir(exist_ok=True, parents=True)
    Path(output_fp).write_text(json.dumps(results, indent=2))
    print(f"Results written to '{output_fp}'")


if __name__ == "__main__":
    typer.run(main)
