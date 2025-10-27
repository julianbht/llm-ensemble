"""Example reader for LLM Judge Challenge 2024 dataset.

Reads the LLM Judge Challenge raw dataset format (queries.txt, documents.jsonl, qrels.txt)
and converts it into normalized JudgingExample records.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from llm_ensemble.ingest.schemas import Query, Document, Relevance, JudgingExample
from llm_ensemble.ingest.ports import ExampleReader


@dataclass(frozen=True)
class LlmJudgePaths:
    """File paths for LLM Judge Challenge 2024 dataset."""

    base_dir: Path

    @property
    def queries(self) -> Path:
        return self.base_dir / "llm4eval_query_2024.txt"

    @property
    def documents(self) -> Path:
        return self.base_dir / "llm4eval_document_2024.jsonl"

    @property
    def qrels(self) -> Path:
        return self.base_dir / "llm4eval_test_qrel_2024.txt"


class LlmJudgeExampleReader(ExampleReader):
    """Reader for LLM Judge Challenge 2024 dataset.

    Reads queries (TSV), documents (JSONL), and relevance judgements (TSV)
    and returns normalized JudgingExample records.

    File format:
    - queries: TSV with columns (query_id, query_text)
    - documents: JSONL with fields {docid, doc}
    - qrels: TSV with columns (query_id, relevance, docid)
    """

    def __init__(self, dataset_id: str = "llm-judge-2024"):
        """Initialize LLM Judge reader.

        Args:
            dataset_id: Dataset identifier for JudgingExample records
        """
        self.dataset_id = dataset_id

    def read(
        self,
        input_path: Path,
        limit: Optional[int] = None,
    ) -> list[JudgingExample]:
        """Read LLM Judge dataset and return JudgingExamples.

        Args:
            input_path: Base directory containing dataset files
            limit: Optional maximum number of examples to return

        Returns:
            List of JudgingExample objects

        Raises:
            FileNotFoundError: If required dataset files are missing
            ValueError: If dataset files are malformed or qrels reference missing queries/documents
        """
        paths = LlmJudgePaths(input_path)

        # Load queries and documents into memory
        queries = self._read_queries(paths.queries)
        docs = self._read_documents(paths.documents)

        # Process qrels and join with queries/documents
        examples = []
        for rel in self._read_qrels(paths.qrels):
            q = queries.get(rel.query_id)
            d = docs.get(rel.docid)

            # Crash if query or document is missing
            if q is None:
                raise ValueError(
                    f"Query '{rel.query_id}' referenced in qrels but not found in queries file"
                )
            if d is None:
                raise ValueError(
                    f"Document '{rel.docid}' referenced in qrels but not found in documents file"
                )

            examples.append(JudgingExample.from_parts(self.dataset_id, q, d, rel))

            # Stop if limit reached
            if limit is not None and len(examples) >= limit:
                break

        return examples

    def _read_queries(self, path: Path) -> Dict[str, Query]:
        """Read TSV of (query_id, query_text) into a dict.

        Args:
            path: Path to queries TSV file

        Returns:
            Dictionary mapping query_id to Query objects

        Raises:
            FileNotFoundError: If queries file doesn't exist
            ValueError: If queries file is malformed
        """
        out: Dict[str, Query] = {}
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.rstrip("\n")
                if not line:
                    continue
                # Expect exactly two columns, split once to be robust to tabs in text
                parts = line.split("\t", maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(f"Invalid query line {i}: {line!r}")
                qid, qtext = parts[0].strip(), parts[1].strip()
                out[qid] = Query(query_id=qid, query_text=qtext)
        return out

    def _read_documents(self, path: Path) -> Dict[str, Document]:
        """Read JSONL of documents into a dict.

        Args:
            path: Path to documents JSONL file

        Returns:
            Dictionary mapping docid to Document objects

        Raises:
            FileNotFoundError: If documents file doesn't exist
            ValueError: If documents file is malformed
        """
        out: Dict[str, Document] = {}
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
                docid = obj.get("docid")
                doc = obj.get("doc")
                if not (isinstance(docid, str) and isinstance(doc, str)):
                    raise ValueError(f"Missing docid/doc at line {i}")
                out[docid] = Document(docid=docid, doc=doc)
        return out

    def _read_qrels(self, path: Path) -> list[Relevance]:
        """Read TSV of qrels and return Relevance objects.

        Args:
            path: Path to qrels TSV file

        Returns:
            List of Relevance objects

        Raises:
            FileNotFoundError: If qrels file doesn't exist
            ValueError: If qrels file is malformed
        """
        qrels = []
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 3:
                    raise ValueError(f"Invalid qrel line {i}: {line!r}")
                qid, rel, docid = parts
                try:
                    rel_i = int(rel)
                except ValueError:
                    raise ValueError(f"Invalid relevance at line {i}: {rel!r}")
                qrels.append(Relevance(query_id=qid, docid=docid, relevance=rel_i))
        return qrels
