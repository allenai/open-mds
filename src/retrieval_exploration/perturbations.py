import math
import random
import warnings
from itertools import zip_longest
from typing import List, Optional, Tuple

import more_itertools
import nlpaug.augmenter.word as naw
import nltk
import numpy as np
import sentence_transformers as st
import torch
from diskcache import Cache
from tqdm import tqdm

from retrieval_exploration.common import util

_SEMANTIC_SIMILARITY_MODEL = "all-MiniLM-L6-v2"
_BT_FROM_MODEL_NAME = "Helsinki-NLP/opus-mt-en-da"
_BT_TO_MODEL_NAME = "Helsinki-NLP/opus-mt-da-en"


class Perturber:
    def __init__(
        self, perturbation: str, doc_sep_token: str, strategy: str = "random", seed: Optional[int] = None
    ) -> None:
        """An object for applying document-level perturbations to some multi-document inputs.

        # Parameters

        perturbation : `str`
            The type of perturbation to apply.
        doc_sep_token : `str`
            The token that separates individual documents in the input strings.
        strategy : `str`, optional (default="random")
            The strategy to use for perturbation. Must be one of `"random"`, `"best-case"`, or `"worst-case"`.
        seed : `int`, optional (default=None)
            If provided, will locally set the seed of the `random` module with this value.

        Usage example:

        >>> inputs = ["document 1 <doc-sep> document 2 <doc-sep> document 3 <doc-sep> document 4"]
        >>> perturber = Perturber("deletion", doc_sep_token="<doc-sep>", strategy="random")
        >>> perturbed_inputs = perturber(inputs, perturbed_frac=0.1)
        """
        self._perturbation = perturbation

        # TODO: Some sort of registry would be better
        perturbation_func = getattr(self, self._perturbation, None)
        if perturbation_func is None:
            raise ValueError(f"Got an unexpected value for perturbation: {perturbation}")
        if strategy not in ["random", "best-case", "worst-case"]:
            raise ValueError(f"Got an unexpected value for strategy: {strategy}")

        self._perturbation_func = perturbation_func
        self._doc_sep_token = doc_sep_token
        self._strategy = strategy
        self._rng = random.Random(seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Non-random strategies need an embedder for document selection
        self._embedder = None
        if self._strategy != "random":
            self._embedder = st.SentenceTransformer(_SEMANTIC_SIMILARITY_MODEL, device=self.device)

        # Some perturbations require special components, like a backtranslation model
        self._aug = None
        if self._perturbation == "backtranslation":
            self._aug = naw.BackTranslationAug(
                from_model_name=_BT_FROM_MODEL_NAME,
                to_model_name=_BT_TO_MODEL_NAME,
                device=self.device,
                # We backtranslate on individual sentences, so this max_length should be plenty.
                max_length=256,
            )

    def __call__(
        self,
        inputs: List[str],
        *,
        perturbed_frac: float = None,
        targets: Optional[List[str]] = None,
        documents: Optional[List[str]] = None,
        unperturbed_indices: Optional[List[int]] = None,
    ) -> List[str]:
        """

        # Parameters

        inputs : `List[str]`
            A list of strings, each string containing the input documents for one example. It is assumed
            that documents are separated by `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            The percentage of documents in each example that should be perturbed. The absolute number of perturbed
            documents will be the ceiling of this value times the original number of documents. Has no effect if
            selected `perturbation` is `"sorting"`. If falsey, returns `inputs` unchanged.
        targets : `List[str]`, optional (default=None)
            If provided, and `strategy` is not `"random"`, the input documents to perturb will be selected based on
            similarity or dissimilarity to these target documents, according to `strategy`. Must be the same
            length as `inputs`.
        documents : `List[str]`, optional (default=None)
            If provided, these documents will be considered (along with the documents in `inputs`) for selection
            during perturbation. Has no effect if selected `perturbation` is not `"addition"` or `"replacement"`.
        """

        if targets is not None and len(inputs) != len(targets):
            raise ValueError(
                "If targets provided, then len(targets) must equal len(inputs)."
                f" Got len(targets)=={len(targets)} and len(inputs)={len(inputs)}."
            )

        if self._perturbation != "sorting" and not perturbed_frac:
            warnings.warn(
                f"perturbed_frac is falsey ({perturbed_frac}) and selected perturbation is not 'sorting'."
                " Inputs will be returned unchanged."
            )
            return inputs

        if documents is not None and self._perturbation not in ["addition", "replacement"]:
            warnings.warn(
                "documents provided, but perturbation is not 'addition' or 'replacement'. They will be ignored."
            )

        # Need an iterable, but an empty list as default value is bad practice
        targets = targets or []

        # All examples that can be considered for selection (ignoring duplicates)
        documents = inputs + documents if documents is not None else inputs
        documents = list(dict.fromkeys(documents))

        # If unperturbed_indices provided, temporarily remove these documents before perturbation...
        if unperturbed_indices is not None:
            inputs, unperturbed_docs, documents = self._remove_unperturbed(
                inputs=inputs, unperturbed_indices=unperturbed_indices, documents=documents
            )

        perturbed_inputs = []
        for example, target in tqdm(
            zip_longest(inputs, targets), desc="Perturbing inputs", total=max(len(inputs), len(targets))
        ):
            perturbed_example = self._perturbation_func(  # type: ignore
                example=example, target=target, perturbed_frac=perturbed_frac, documents=documents
            )
            perturbed_inputs.append(perturbed_example)

        # ... then, insert them back in their original positions after perturbation
        if unperturbed_indices is not None:
            perturbed_inputs = self._replace_unperturbed(
                perturbed_inputs=perturbed_inputs,
                unperturbed_docs=unperturbed_docs,
                unperturbed_indices=unperturbed_indices,
            )

        return perturbed_inputs

    def backtranslation(
        self,
        *,
        example: str,
        perturbed_frac: float,
        target: Optional[str] = None,
        documents: Optional[List[str]] = None,
    ) -> str:
        """ """

        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)
        num_docs = util.get_num_docs(example, doc_sep_token=self._doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * num_docs)

        # If we are backtranslating all documents, we do not need to sample
        if k == num_docs:
            sampled_docs = input_docs
        elif self._strategy == "random":
            sampled_docs = self._select_docs([example], k=k)
        else:
            sampled_docs = self._select_docs(
                documents=[example],
                k=k,
                target=target,
                largest=self._strategy == "worst-case",
            )

        # Back translate the sampled documents. To save computation, cache the backtranslation results to disk
        back_translated_docs = self._get_backtranslated_docs(sampled_docs)

        for sampled, translated in zip(sampled_docs, back_translated_docs):
            input_docs[input_docs.index(sampled)] = " ".join(sent.strip() for sent in translated)

        perturbed_example = f" {self._doc_sep_token} ".join(input_docs)
        return perturbed_example

    def sorting(
        self,
        *,
        example: str,
        perturbed_frac: Optional[float] = None,
        target: Optional[str] = None,
        documents: Optional[List[str]] = None,
    ) -> str:
        """Perturbs the documents in `example` by sorting them according to the selected `strategy`.

        # Parameters

        example : `str`
            A string containing the input documents for one example. It is assumed that documents are separated by
            `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            Has no effect. Exists for consistency with other perturbation functions.
        target : `str`, optional (default=None)
            If provided, documents will be perturbed based on comparison to this text.
        documents : `List[str]`, optional (default=None)
            Has no effect. Exists for consistency with other perturbation functions.
        """
        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)

        if self._strategy == "random":
            self._rng.shuffle(input_docs)
        else:
            input_docs = self._select_docs(
                documents=[example],
                k=len(input_docs),
                target=target,
                largest=self._strategy == "best-case",
            )

        perturbed_example = f" {self._doc_sep_token} ".join(input_docs)
        return perturbed_example

    def duplication(
        self,
        *,
        example: str,
        perturbed_frac: float,
        target: Optional[str] = None,
        documents: Optional[List[str]] = None,
    ) -> str:
        """Perturbs the documents in `example` by duplicating them according to the selected `strategy`.

        # Parameters

        example : `str`
            A string containing the input documents for one example. It is assumed that documents are separated by
            `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            The percentage of documents in each example that should be perturbed. The absolute number of perturbed
            documents will be the ceiling of this value times the original number of documents.
        target : `str`, optional (default=None)
            If provided, documents will be perturbed based on comparison to this text.
        documents : `List[str]`, optional (default=None)
            Has no effect. Exists for consistency with other perturbation functions.
        """
        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)
        num_docs = util.get_num_docs(example, doc_sep_token=self._doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * num_docs)

        # If we are duplicating all documents, we do not need to sample
        if k == num_docs:
            repeaters = input_docs
        elif self._strategy == "random":
            repeaters = self._rng.sample(input_docs, k)
        else:
            repeaters = self._select_docs(
                documents=[example],
                k=k,
                target=target,
                largest=self._strategy == "best-case",
            )

        perturbed_example = f" {self._doc_sep_token} ".join(input_docs + repeaters)
        return perturbed_example

    def addition(
        self,
        *,
        example: str,
        perturbed_frac: float,
        documents: List[str],
        target: Optional[str] = None,
    ) -> str:
        """Perturbs the documents in `example` by adding additional documents according to the selected `strategy`.

        # Parameters

        example : `str`
            A string containing the input documents for one example. It is assumed that documents are separated by
            `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            The percentage of documents in each example that should be perturbed. The absolute number of perturbed
            documents will be the ceiling of this value times the original number of documents.
        target : `str`, optional (default=None)
            If provided, documents will be perturbed based on comparison to this text.
        documents : `List[str]`, optional (default=None)
            If provided, these documents will be considered (along with the documents in `example`) for selection
            during perturbation.
        """
        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)
        num_docs = util.get_num_docs(example, doc_sep_token=self._doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * num_docs)

        if self._strategy == "random":
            sampled_docs = self._select_docs(documents, k=k, query=example)
        else:
            sampled_docs = self._select_docs(
                documents=documents,
                k=k,
                query=example,
                target=target,
                largest=self._strategy == "best-case",
            )

        perturbed_example = f" {self._doc_sep_token} ".join(input_docs + sampled_docs)
        return perturbed_example

    def deletion(
        self,
        *,
        example: str,
        perturbed_frac: float,
        target: Optional[str] = None,
        documents: Optional[List[str]] = None,
    ) -> str:
        """Perturbs the documents in `example` by deleting documents according to the selected `strategy`.

        # Parameters

        example : `str`
            A string containing the input documents for one example. It is assumed that documents are separated by
            `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            The percentage of documents in each example that should be perturbed. The absolute number of perturbed
            documents will be the ceiling of this value times the original number of documents.
        target : `str`, optional (default=None)
            If provided, documents will be perturbed based on comparison to this text.
        documents : `List[str]`, optional (default=None)
            Has no effect. Exists for consistency with other perturbation functions.
        """
        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)
        num_docs = util.get_num_docs(example, doc_sep_token=self._doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * num_docs)

        # If we are deleting all documents, just return the empty string.
        if k == num_docs:
            return ""

        if self._strategy == "random":
            to_delete = self._rng.sample(range(num_docs), k)
        else:
            sampled_docs = self._select_docs(
                documents=[example],
                k=k,
                target=target,
                largest=self._strategy == "worst-case",
            )
            to_delete = [input_docs.index(doc) for doc in sampled_docs]

        # Collect the perturbed example
        pertured_example = f" {self._doc_sep_token} ".join(
            doc for j, doc in enumerate(input_docs) if j not in to_delete
        )
        return pertured_example

    def replacement(
        self,
        *,
        example: str,
        perturbed_frac: float,
        documents: List[str],
        target: Optional[str] = None,
    ) -> str:
        """Perturbs the documents in `example` by replacing them according to the selected `strategy`.

        # Parameters

        example : `str`
            A string containing the input documents for one example. It is assumed that documents are separated by
            `doc_sep_token`.
        perturbed_frac : `float`, optional (default=None)
            The percentage of documents in each example that should be perturbed. The absolute number of perturbed
            documents will be the ceiling of this value times the original number of documents.
        target : `str`, optional (default=None)
            If provided, documents will be perturbed based on comparison to this text.
        documents : `List[str]`, optional (default=None)
            If provided, these documents will be considered (along with the documents in `example`) for selection
            during perturbation.
        """
        input_docs = util.split_docs(example, doc_sep_token=self._doc_sep_token)
        num_docs = util.get_num_docs(example, doc_sep_token=self._doc_sep_token)

        # The absolute number of documents to perturb
        k = math.ceil(perturbed_frac * num_docs)

        to_replace = input_docs if k == num_docs else None

        if self._strategy == "random":
            sampled_docs = self._select_docs(documents, k=k, query=example)
            to_replace = to_replace or self._select_docs([example], k=k)

        else:
            # In the best case, replace the least similar documents with the most similar documents and vice versa
            # in the worst case.
            largest = self._strategy == "best-case"
            sampled_docs = self._select_docs(
                documents=documents,
                k=k,
                query=example,
                target=target,
                largest=largest,
            )
            to_replace = to_replace or self._select_docs(
                documents=[example],
                k=k,
                target=target,
                largest=not largest,
            )
        replace_indices = [input_docs.index(doc) for doc in to_replace]

        for i, doc in zip(replace_indices, sampled_docs):
            input_docs[i] = doc.strip()

        perturbed_example = f" {self._doc_sep_token} ".join(input_docs)
        return perturbed_example

    def _select_docs(
        self,
        documents: List[str],
        *,
        k: int,
        query: Optional[str] = None,
        target: Optional[str] = None,
        largest: bool = True,
    ) -> List[str]:
        """Randomly samples `k` documents without replacement from `documents` according to `strategy`. Assumes
        that each string in `documents` contains one or more documents separated by `doc_sep_token`. Any documents
        in `query`, which should be formatted similar to documents, will be excluded from selection.

        # Parameters

        documents : `List[str]`
            A list of strings to select documents from. It is assumed that each string contains the input documents
            for one example and that items in this list are separated by `doc_sep_token`.
        k : `int`
            The number of documents to sample (without replacement) from `documents`.
        query : `str`, optional (default=None)
            If provided, documents in `query` will not be sampled from `documents`. Documents will be selected
            based on similarity or dissimilarity to these documents, according to `strategy`. Should be provided
            in the same format as `documents`.
        target : `str`, optional (default=None)
            If provided, documents will be selected based on comparison to `target` instead of `query`.
        largest : `bool`
            If `True`, the top-k documents are returned. Otherwise the bottom-k documents are returned.
        """
        if self._strategy != "random" and not query and not target:
            raise ValueError(
                "Must provide either a `query` or a `target` when using a `strategy` other than `random`."
            )
        if self._strategy == "random" and target is not None:
            warnings.warn("strategy is random, but target is not None. target will be ignored.")

        # Extract all individual documents
        documents = list(
            more_itertools.flatten(
                util.split_docs(example, doc_sep_token=self._doc_sep_token) for example in documents
            )
        )

        # If query is provided, remove it from the possible inputs
        if query is not None:
            query_docs = util.split_docs(query, doc_sep_token=self._doc_sep_token)
            documents = [doc for doc in documents if doc not in query_docs]

        # Check that we have enough documents to sample from
        if len(documents) < k:
            raise ValueError(
                f"Not enough unique documents to sample {k} without replacement. Only have {len(documents)}."
            )

        if self._strategy == "random":
            return self._rng.sample(documents, k)

        # Cache all inputs document embeddings to make this as fast as possible.
        doc_embeddings = self._get_doc_embeddings(documents)

        # If target is provided, look for docs most similar to it. Otherwise look for docs most similar to the query.
        if target:
            target_embedding = self._embedder.encode(  # type: ignore
                target, convert_to_tensor=True, device=self.device, normalize_embeddings=True
            )
            scores = st.util.dot_score(target_embedding, doc_embeddings)[0]
        else:
            query_embeddings = self._embedder.encode(  # type: ignore
                query_docs, convert_to_tensor=True, device=self.device, normalize_embeddings=True
            )
            scores = st.util.dot_score(query_embeddings, doc_embeddings)
            scores = torch.mean(scores, axis=0)

        # Return the top k most similar (or dissimilar) documents
        indices = torch.topk(scores, k=k, largest=largest, sorted=True).indices
        return [documents[i] for i in indices]

    def _remove_unperturbed(
        self, inputs: List[str], unperturbed_indices: List[int], documents: Optional[List[str]] = None
    ) -> Tuple[List[str], List[List[str]], Optional[List[str]]]:
        """Given a list of `unperturbed_indices`, remove the corresponding documents from `inputs` and `documents` and
        returns a tuple of the resulting `inputs`, `unperturbed_indices` and `documents`.
        """
        example_docs = [util.split_docs(example, doc_sep_token=self._doc_sep_token) for example in inputs]
        unperturbed_docs = [[docs[i] for i in unperturbed_indices] for docs in example_docs]
        example_docs = [
            [doc for doc in example if doc not in more_itertools.flatten(unperturbed_docs)] for example in example_docs
        ]
        inputs = [f" {self._doc_sep_token} ".join(docs) for docs in example_docs]
        # Horribly complicated way to remove all the unperturbed_docs from documents
        if documents is not None:
            documents = list(
                set(
                    more_itertools.flatten(
                        util.split_docs(example, doc_sep_token=self._doc_sep_token) for example in documents
                    )
                )
                - set(more_itertools.flatten(unperturbed_docs))
            )
        return inputs, unperturbed_docs, documents

    def _replace_unperturbed(
        self, perturbed_inputs: List[str], unperturbed_docs: List[List[str]], unperturbed_indices: List[int]
    ) -> List[str]:
        """Inserts `unperturbed_docs` into `perturbed_inputs` at `unperturbed_indices`"""
        perturbed_docs = [util.split_docs(example, doc_sep_token=self._doc_sep_token) for example in perturbed_inputs]
        for i, (unperturbed_example, perturbed_example) in enumerate(zip(unperturbed_docs, perturbed_docs)):
            for unperturbed_doc, unperturbed_idx in zip(unperturbed_example, unperturbed_indices):
                perturbed_example.insert(unperturbed_idx, unperturbed_doc)
            perturbed_inputs[i] = f" {self._doc_sep_token} ".join(perturbed_example)
        return perturbed_inputs

    def _get_doc_embeddings(self, documents: List[str]) -> torch.Tensor:
        doc_embeddings = []

        with Cache(util.CACHE_DIR) as reference:
            for doc in documents:
                key = f"{_SEMANTIC_SIMILARITY_MODEL}_{util.sanitize_text(doc, lowercase=True)}"
                if key in reference:
                    doc_embeddings.append(reference[key])
                else:
                    embedding = self._embedder.encode(  # type: ignore
                        doc, convert_to_numpy=True, device=self.device, normalize_embeddings=True
                    )
                    doc_embeddings.append(embedding)
                    reference[key] = embedding

        # Converting a list of numpy arrays to a numpy array before the call to as_tensor is significantly faster.
        doc_embeddings = torch.as_tensor(np.array(doc_embeddings), device=self.device)  # type: ignore

        return doc_embeddings

    def _get_backtranslated_docs(self, documents: List[str]) -> List[str]:
        back_translated_docs = []

        with Cache(util.CACHE_DIR) as reference:
            for doc in documents:
                key = f"{_BT_FROM_MODEL_NAME}_{_BT_TO_MODEL_NAME}_{util.sanitize_text(doc, lowercase=True)}"
                if key in reference:
                    back_translated_docs.append(reference[key])
                else:
                    # We backtranslate individual sentences, which improves backtranslation quality.
                    # This is likely because it more closely matches the MT models training data.
                    back_translated_sents = self._aug.augment(nltk.sent_tokenize(doc))  # type: ignore
                    back_translated_doc = util.sanitize_text(" ".join(sent for sent in back_translated_sents))
                    back_translated_docs.append(back_translated_doc)
                    reference[key] = back_translated_doc

        return back_translated_docs
