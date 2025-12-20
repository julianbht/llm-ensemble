"""Thomas et al. simple prompt builder adapter.

Simple prompt builder using string formatting.
All template metadata owned by this adapter as class constants.
"""

from __future__ import annotations
import uuid
from textwrap import dedent

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.application.ports.driven.prompt_builder_port import PromptBuilderPort
from llm_ensemble.infer.domain.entities.prompt_builder import PromptBuilder


class ThomasSimplePromptBuilder(PromptBuilderPort):
    """Thomas et al. simple prompt (3-point relevance scoring).

    Template substitutes:
    - {query} - The query text
    - {document} - The document text
    """

    TEMPLATE_NAME = "thomas-simple"
    TEMPLATE_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "thomas-simple-prompt-builder-v1")
    TEMPLATE_TEXT = dedent("""\
        Given a query and a web page, you must
        provide a score on an integer scale of 0 to 2 with the
        following meanings:
        2 = highly relevant, very helpful for this query
        1 = relevant, may be partly helpful but might contain
        other irrelevant content
        0 = not relevant, should never be shown for this query
        Assume that you are writing a report on the subject of the
        topic. If you would use any of the information contained
        in the web page in such a report, mark it 1. If the web page
        is primarily about the topic, or contains vital information
        about the topic, mark it 2. Otherwise, mark it 0.

        Query
        A person has typed [{query}] into a search engine.

        Result
        Consider the following web page.
        —BEGIN WEB PAGE CONTENT—
        {document}
        —END WEB PAGE CONTENT—

        Instructions
        Split this problem into steps:
        Consider the underlying intent of the search.
        Decide on a final relevance score for the page (O) without providing
        any reasoning. Example: {{"O": 2}}""")

    def __init__(self):
        """Initialize builder and create cached PromptBuilder entity."""
        self._builder = PromptBuilder(
            id=self.TEMPLATE_ID,
            name=self.TEMPLATE_NAME,
            version="1.0"
        )

    def build_prompt(self, dataset_sample: DatasetSample) -> str:
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

    def get_template_text(self) -> str:
        """Get the raw template text for this builder.

        Returns:
            Raw template string (unrendered)
        """
        return self.TEMPLATE_TEXT

    def _render(self, dataset_sample: DatasetSample) -> str:
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
