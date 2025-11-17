"""Dataset reader for LLM Judge Challenge 2024 dataset.

Reads the LLM Judge Challenge raw dataset format (queries.txt, documents.jsonl, qrels.txt)
and returns a complete NormalizedDataset.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from llm_ensemble.ingest.schemas import Query, Document, RelevanceScore, Dataset, JudgingSample, NormalizedDataset
from llm_ensemble.ingest.ports import DatasetReader


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
        """Prefer qrels file that actually includes relevance labels."""
        dev_path = self.base_dir / "llm4eval_dev_qrel_2024.txt"
        if dev_path.exists():
            return dev_path
        return self.base_dir / "llm4eval_test_qrel_2024.txt"


class LLMJudgeDatasetReader(DatasetReader):
    """Reader for LLM Judge Challenge 2024 dataset.

    Reads queries (TSV), documents (JSONL), and relevance judgements (TSV)
    and returns dataset metadata + samples.

    File format:
    - queries: TSV with columns (query_id, query_text)
    - documents: JSONL with fields {docid, doc}
    - qrels: TSV with columns (query_id, relevance, docid) or (query_id, iteration, docid, relevance)
    """

    def read(
        self,
        input_path: str,
        limit: Optional[int] = None,
    ) -> NormalizedDataset:
        """Read and normalize LLM Judge dataset.

        Args:
            input_path: Base directory containing dataset files (as string path)
            limit: Optional maximum number of samples to return

        Returns:
            NormalizedDataset with complete samples

        Raises:
            FileNotFoundError: If required dataset files are missing
            ValueError: If dataset files are malformed or qrels reference missing queries/documents
        """
        paths = LlmJudgePaths(Path(input_path))

        # Create Dataset entity (dataset metadata extracted from data context)
        dataset = Dataset.create("llmjudge", description="LLM Judge Challenge 2024")

        # Load queries and documents into memory (with IDs computed from dataset)
        queries = self._read_queries(paths.queries, dataset)
        docs = self._read_documents(paths.documents, dataset)

        # Process qrels and join with queries/documents
        samples = []
        for qid, docid, relevance in self._read_qrels(paths.qrels):
            q = queries.get(qid)
            d = docs.get(docid)

            # Crash if query or document is missing
            if q is None:
                raise ValueError(
                    f"Query '{qid}' referenced in qrels but not found in queries file"
                )
            if d is None:
                raise ValueError(
                    f"Document '{docid}' referenced in qrels but not found in documents file"
                )

            # Create complete JudgingSample (reader does full normalization)
            sample = JudgingSample.create(
                query=q,
                document=d,
                gold_score=RelevanceScore(relevance),
            )
            samples.append(sample)

            # Stop if limit reached
            if limit is not None and len(samples) >= limit:
                break

        return NormalizedDataset(dataset=dataset, samples=samples)

    def _read_queries(self, path: Path, dataset: Dataset) -> Dict[str, Query]:
        """Read TSV of (query_id, query_text) into a dict.

        Args:
            path: Path to queries TSV file
            dataset: Dataset entity

        Returns:
            Dictionary mapping query_id to Query objects (with IDs computed)

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
                out[qid] = Query.create(dataset, qid, qtext)
        return out

    def _read_documents(self, path: Path, dataset: Dataset) -> Dict[str, Document]:
        """Read JSONL of documents into a dict.

        Args:
            path: Path to documents JSONL file
            dataset: Dataset entity

        Returns:
            Dictionary mapping docid to Document objects (with IDs computed)

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
                out[docid] = Document.create(dataset, docid, doc)
        return out

    def _read_qrels(self, path: Path) -> list[tuple[str, str, int]]:
        """Read TSV of qrels and return tuples of (query_id, docid, relevance).

        Args:
            path: Path to qrels TSV file

        Returns:
            List of (query_id, docid, relevance) tuples

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
                if len(parts) == 3:
                    qid, rel, docid = parts
                elif len(parts) == 4:
                    qid, _, docid, rel = parts
                else:
                    raise ValueError(f"Invalid qrel line {i}: {line!r}")
                try:
                    rel_i = int(rel)
                except ValueError:
                    raise ValueError(f"Invalid relevance at line {i}: {rel!r}")
                qrels.append((qid, docid, rel_i))
        return qrels
