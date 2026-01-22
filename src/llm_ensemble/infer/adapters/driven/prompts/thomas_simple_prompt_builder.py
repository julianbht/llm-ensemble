"""Thomas et al. simple prompt builder adapter.

Simple prompt builder that asks for a single relevance score.
All template metadata owned by this adapter as class constants.
"""

from __future__ import annotations
import uuid
from textwrap import dedent

from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.infer.application.ports.driven.for_building_prompts import ForBuildingPrompts
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder


class ThomasSimplePromptBuilder(ForBuildingPrompts):
    """Thomas et al. simple prompt asking for a single relevance score.

    A straightforward prompt that:
    - Defines the 4-level relevance scale (0-3)
    - Presents the query and document
    - Asks for a single numeric score

    Template substitutes:
    - {query} - The query text
    - {document} - The document text
    """

    TEMPLATE_NAME = "thomas-simple"
    TEMPLATE_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.thomas-simple-v1")
    TEMPLATE_TEXT = dedent("""\
        Given a query and a document, provide a relevance score on a scale of 0 to 3:

        3 = Perfectly relevant: The document is dedicated to the query and contains the exact answer.
        2 = Highly relevant: The document has some answer for the query, but it may be unclear or hidden amongst other information.
        1 = Related: The document seems related to the query but does not answer it.
        0 = Irrelevant: The document has nothing to do with the query.

        Query: {query}

        Document:
        {document}

        Respond with only a JSON object containing the score.
        Example: {{"score": 2}}""")

    def __init__(self):
        """Initialize builder and create cached PromptBuilder entity."""
        self._builder = PromptBuilder(
            id=self.TEMPLATE_ID,
            name=self.TEMPLATE_NAME,
            version="1.0"
        )

    def build_prompt(self, dataset_sample: NormalizedDatasetJudgingSample) -> str:
        """Render prompt text from dataset sample.

        Renders the prompt text using the internal template.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Rendered prompt text ready for inference

        Raises:
            KeyError: If template variables missing from dataset
        """
        return self._render(dataset_sample)

    def get_builder(self) -> PromptBuilder:
        """Get PromptBuilder metadata for this adapter.

        Returns:
            PromptBuilder entity with id, name, and version
        """
        return self._builder

    def _render(self, dataset_sample: NormalizedDatasetJudgingSample) -> str:
        """Render prompt text from dataset sample.

        Pure, testable rendering function with no domain dependencies.
        Extracts query and document text from dataset_sample and
        substitutes into template using string formatting.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            Rendered prompt text string

        Raises:
            KeyError: If template variables missing from dataset
        """
        judging_sample = dataset_sample.judging_sample
        query_text = judging_sample.query.query_text
        document_text = judging_sample.document.doc_text

        return self.TEMPLATE_TEXT.format(
            query=query_text,
            document=document_text
        )

    def get_template_text(self) -> str:
        """Get the raw template text for this builder.

        Returns:
            Raw template string (unrendered)
        """
        return self.TEMPLATE_TEXT
