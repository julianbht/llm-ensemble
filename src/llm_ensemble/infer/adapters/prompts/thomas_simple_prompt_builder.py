"""Thomas et al. simple prompt builder adapter.

Simple prompt builder using string formatting.
All template metadata owned by this adapter as class constants.
"""

from __future__ import annotations
import uuid
from textwrap import dedent

from llm_ensemble.ingest.schemas.dataset_sample import DatasetSample
from llm_ensemble.infer.ports import PromptBuilderPort
from llm_ensemble.infer.schemas.entities.prompt_template import PromptTemplate
from llm_ensemble.infer.schemas.entities.llm_prompt import LLMPrompt


class ThomasSimplePromptBuilder(PromptBuilderPort):
    """Thomas et al. simple prompt (3-point relevance scoring).

    Template substitutes:
    - {query} - The query text
    - {document} - The document text
    """

    TEMPLATE_NAME = "thomas-simple"
    TEMPLATE_ID = uuid.uuid5(uuid.NAMESPACE_DNS, "llm-ensemble.thomas-simple-v1")
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
        """Initialize builder and create cached template entity."""
        self._template = PromptTemplate(
            id=self.TEMPLATE_ID,
            name=self.TEMPLATE_NAME,
            template_text=self.TEMPLATE_TEXT
        )

    def build_prompt(self, dataset_sample: DatasetSample) -> LLMPrompt:
        """Build complete LLMPrompt entity from dataset sample.

        Renders the prompt text and constructs an LLMPrompt domain entity
        with template metadata, sample context, and rendered text.

        Args:
            dataset_sample: DatasetSample containing judging_sample and context

        Returns:
            LLMPrompt domain entity ready for inference

        Raises:
            KeyError: If template variables missing from dataset
        """
        prompt_text = self._render(dataset_sample)
        return LLMPrompt(
            prompt_template=self._template,
            dataset_sample=dataset_sample,
            prompt_text=prompt_text
        )

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
