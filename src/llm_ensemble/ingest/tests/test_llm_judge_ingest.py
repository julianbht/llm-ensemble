"""Tests for LLM Judge dataset ingestion.

These tests verify the adapter layer can correctly parse and normalize
the LLM Judge Challenge dataset into canonical JudgingExample records.
"""

import json
import pytest
from pathlib import Path

from llm_ensemble.ingest.adapters.llm_judge import (
    iter_examples,
    read_queries,
    read_documents,
    read_qrels,
)
from llm_ensemble.ingest.domain.models import JudgingExample


def _write(tmp: Path, name: str, content: str) -> Path:
    """Helper to write test fixture files."""
    p = tmp / name
    p.write_text(content, encoding="utf-8")
    return p


class TestIterExamples:
    """Test the main iter_examples function that orchestrates parsing."""

    def test_basic_example(self, tmp_path: Path):
        """Test parsing a single complete query-doc-qrel triplet."""
        _write(tmp_path, "llm4eval_query_2024.txt", "q1\tWhat is AI?\n")
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "AI is..."}) + "\n",
        )
        _write(tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\n")

        items = list(iter_examples(tmp_path))
        assert len(items) == 1
        ex = items[0]
        assert ex.query_id == "q1"
        assert ex.query_text == "What is AI?"
        assert ex.docid == "d1"
        assert ex.doc == "AI is..."
        assert ex.gold_relevance == 1
        assert ex.dataset == "llm-judge-2024"

    def test_multiple_examples(self, tmp_path: Path):
        """Test parsing multiple query-document pairs."""
        _write(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tWhat is AI?\nq2\tDefine machine learning\n",
        )
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "AI is..."}) + "\n"
            + json.dumps({"docid": "d2", "doc": "ML is..."}) + "\n",
        )
        _write(
            tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\nq2 0 d2\n"
        )

        items = list(iter_examples(tmp_path))
        assert len(items) == 2
        assert items[0].query_id == "q1"
        assert items[0].gold_relevance == 1
        assert items[1].query_id == "q2"
        assert items[1].gold_relevance == 0

    def test_missing_query_skips_qrel(self, tmp_path: Path, capsys):
        """Test that qrels with missing queries are skipped with warning."""
        _write(tmp_path, "llm4eval_query_2024.txt", "q1\tQuery one\n")
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Doc 2"}) + "\n",
        )
        # q2 exists in qrel but not in queries file
        _write(
            tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\nq2 1 d2\n"
        )

        items = list(iter_examples(tmp_path))
        assert len(items) == 1  # only q1 processed
        captured = capsys.readouterr()
        assert "Skipped 1 qrels with missing query" in captured.out

    def test_missing_document_skips_qrel(self, tmp_path: Path, capsys):
        """Test that qrels with missing documents are skipped with warning."""
        _write(
            tmp_path,
            "llm4eval_query_2024.txt",
            "q1\tQuery one\nq2\tQuery two\n",
        )
        _write(
            tmp_path,
            "llm4eval_document_2024.jsonl",
            json.dumps({"docid": "d1", "doc": "Doc 1"}) + "\n",
        )
        # d2 exists in qrel but not in documents file
        _write(
            tmp_path, "llm4eval_test_qrel_2024.txt", "q1 1 d1\nq2 1 d2\n"
        )

        items = list(iter_examples(tmp_path))
        assert len(items) == 1
        captured = capsys.readouterr()
        assert "Skipped 0 qrels with missing query" in captured.out
        assert "1 with missing doc" in captured.out


class TestReadQueries:
    """Test the query file parser."""

    def test_parse_valid_queries(self, tmp_path: Path):
        """Test parsing well-formed TSV queries."""
        qfile = _write(
            tmp_path, "queries.txt", "q1\tWhat is AI?\nq2\tDefine ML\n"
        )
        queries = read_queries(qfile)
        assert len(queries) == 2
        assert queries["q1"].query_text == "What is AI?"
        assert queries["q2"].query_text == "Define ML"

    def test_skip_empty_lines(self, tmp_path: Path):
        """Test that empty lines are ignored."""
        qfile = _write(
            tmp_path, "queries.txt", "q1\tQuery one\n\nq2\tQuery two\n"
        )
        queries = read_queries(qfile)
        assert len(queries) == 2

    def test_query_with_tab_in_text(self, tmp_path: Path):
        """Test queries containing tabs in the query text itself."""
        qfile = _write(
            tmp_path, "queries.txt", "q1\tWhat is\tAI and ML?\n"
        )
        queries = read_queries(qfile)
        assert queries["q1"].query_text == "What is\tAI and ML?"

    def test_malformed_query_line_raises(self, tmp_path: Path):
        """Test that lines without exactly 2 columns raise an error."""
        qfile = _write(tmp_path, "queries.txt", "q1_missing_query\n")
        with pytest.raises(ValueError, match="Invalid query line"):
            read_queries(qfile)


class TestReadDocuments:
    """Test the document JSONL parser."""

    def test_parse_valid_documents(self, tmp_path: Path):
        """Test parsing well-formed JSONL documents."""
        dfile = _write(
            tmp_path,
            "docs.jsonl",
            json.dumps({"docid": "d1", "doc": "Text 1"}) + "\n"
            + json.dumps({"docid": "d2", "doc": "Text 2"}) + "\n",
        )
        docs = read_documents(dfile)
        assert len(docs) == 2
        assert docs["d1"].doc == "Text 1"
        assert docs["d2"].doc == "Text 2"

    def test_skip_empty_lines(self, tmp_path: Path):
        """Test that empty/whitespace lines are skipped."""
        dfile = _write(
            tmp_path,
            "docs.jsonl",
            json.dumps({"docid": "d1", "doc": "Text"}) + "\n\n  \n",
        )
        docs = read_documents(dfile)
        assert len(docs) == 1

    def test_invalid_json_raises(self, tmp_path: Path):
        """Test that malformed JSON raises a descriptive error."""
        dfile = _write(tmp_path, "docs.jsonl", "{not valid json}\n")
        with pytest.raises(ValueError, match="Invalid JSONL at line 1"):
            read_documents(dfile)

    def test_missing_docid_raises(self, tmp_path: Path):
        """Test that missing docid field raises an error."""
        dfile = _write(
            tmp_path, "docs.jsonl", json.dumps({"doc": "Text"}) + "\n"
        )
        with pytest.raises(ValueError, match="Missing docid/doc at line 1"):
            read_documents(dfile)

    def test_missing_doc_raises(self, tmp_path: Path):
        """Test that missing doc field raises an error."""
        dfile = _write(
            tmp_path, "docs.jsonl", json.dumps({"docid": "d1"}) + "\n"
        )
        with pytest.raises(ValueError, match="Missing docid/doc at line 1"):
            read_documents(dfile)


class TestReadQrels:
    """Test the qrel (query-document relevance) parser."""

    def test_parse_valid_qrels(self, tmp_path: Path):
        """Test parsing well-formed qrel lines."""
        qfile = _write(tmp_path, "qrels.txt", "q1 1 d1\nq2 0 d2\n")
        qrels = list(read_qrels(qfile))
        assert len(qrels) == 2
        assert qrels[0].query_id == "q1"
        assert qrels[0].relevance == 1
        assert qrels[0].docid == "d1"
        assert qrels[1].relevance == 0

    def test_skip_empty_lines(self, tmp_path: Path):
        """Test that empty lines are ignored."""
        qfile = _write(tmp_path, "qrels.txt", "q1 1 d1\n\nq2 0 d2\n")
        qrels = list(read_qrels(qfile))
        assert len(qrels) == 2

    def test_invalid_format_raises(self, tmp_path: Path):
        """Test that lines without 3 columns raise an error."""
        qfile = _write(tmp_path, "qrels.txt", "q1 1\n")
        with pytest.raises(ValueError, match="Invalid qrel line 1"):
            list(read_qrels(qfile))

    def test_non_integer_relevance_raises(self, tmp_path: Path):
        """Test that non-numeric relevance raises an error."""
        qfile = _write(tmp_path, "qrels.txt", "q1 bad d1\n")
        with pytest.raises(ValueError, match="Invalid relevance at line 1"):
            list(read_qrels(qfile))