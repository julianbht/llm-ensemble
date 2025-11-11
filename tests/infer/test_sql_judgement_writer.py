"""Tests for SqlJudgementWriter.

Simple tests verifying the SQL writer can decompose and persist judgements.
Uses in-memory SQLite for fast, isolated testing.
"""

import pytest
from pathlib import Path
from uuid import uuid4

from llm_ensemble.infer.adapters.io.sql_judgement_writer import SqlJudgementWriter
from llm_ensemble.infer.schemas.llm_judgement import LLMJudgement
from llm_ensemble.infer.schemas.llm_judgement import LLMRequest
from llm_ensemble.infer.schemas.llm_judgement import LLMResponse
from llm_ensemble.infer.schemas.llm_judgement import LLMScore
from llm_ensemble.infer.schemas.infer_run_info import InferRunInfo
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.ingest.schemas.judging_sample import JudgingSample
from llm_ensemble.ingest.schemas import Query, Document, Dataset, IngestRunInfo
from llm_ensemble.libs.schemas import IOConfig, RelevanceScore
from llm_ensemble.libs.runtime.run_info import RunType
from llm_ensemble.libs.db import Base, create_all_tables
from sqlalchemy import create_engine
import os


@pytest.fixture
def in_memory_db():
    """Create in-memory SQLite database with schema."""
    # Use in-memory SQLite (no .env needed)
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"

    engine = create_engine("sqlite:///:memory:")
    create_all_tables(engine)
    yield engine

    # Cleanup
    if "DATABASE_URL" in os.environ:
        del os.environ["DATABASE_URL"]


@pytest.fixture
def sample_judgement():
    """Create a minimal LLMJudgement for testing."""
    # Create dataset and sample
    dataset = Dataset.create("test-dataset", "Test dataset")
    query = Query.create(dataset, "q1", "What is Python?")
    doc = Document.create(dataset, "d1", "Python is a programming language")

    ingest_run_info = IngestRunInfo.create(
        run_name="ingest_test",
        io_config_name="test",
        io_config=IOConfig(name_hint="test", description="test", reader_module="test", reader_class="Test", writer_module="test", writer_class="Test"),
        input_path="/test",
        limit=None,
        run_type=RunType.TEST,
        notes=None,
        git_sha="abc123",
        git_clean=True,
        git_branch="main",
    )

    sample = JudgingSample.create(query, doc, RelevanceScore.RELEVANT, ingest_run_info)

    # Create model config
    model_cfg = ModelConfig(
        name_hint="test-model",
        model_id="test-model-id",
        provider="openrouter",
        provider_module="test.provider",
        provider_class="TestProvider",
        context_window=4096,
        temperature=0.7,
    )

    # Create prompt config
    prompt_cfg = PromptConfig(
        name_hint="test-prompt",
        description="Test prompt",
        builder_module="llm_ensemble.infer.adapters.prompts.thomas_builder_simple",
        builder_class="JinjaPromptBuilder",
        parser_module="llm_ensemble.infer.adapters.parsers.json_response_parser",
        parser_class="JsonResponseParser",
    )

    # Create run info
    run_info = InferRunInfo(
        run_name="infer_test_run",
        run_type=RunType.TEST,
        notes=None,
        git_sha="abc123",
        git_clean=True,
        git_branch="main",
        model_config_name="test-model",
        prompt_config_name="test-prompt",
        io_config_name="test-io",
        model_cfg=model_cfg,
        prompt_config=prompt_cfg,
        io_config=IOConfig(name_hint="test", description="test", reader_module="test", reader_class="Test", writer_module="test", writer_class="Test"),
        input_file="/test/input.ndjson",
        limit=None,
    )

    # Create judgement
    request = LLMRequest(prompt="Test prompt", warnings=[])
    response = LLMResponse(raw_response='{"O": 1}', latency_ms=100.0, retries=0, warnings=[])
    score = LLMScore(label=RelevanceScore.RELEVANT, confidence=0.9, rationale="Test rationale", warnings=[])

    return LLMJudgement(
        judging_sample=sample,
        llm_request=request,
        llm_response=response,
        llm_score=score,
        run_info=run_info,
    )


@pytest.mark.integration
def test_sql_writer_writes_judgement(in_memory_db, sample_judgement, tmp_path):
    """Test that SqlJudgementWriter can write a judgement to database."""
    writer = SqlJudgementWriter()

    with writer.open(tmp_path) as w:
        result = w.write_one(sample_judgement)
        assert result.item_type == "llm_call"
        assert result.item_id is not None

    summary = writer.get_summary()
    assert summary.judgements_written == 1


@pytest.mark.integration
def test_sql_writer_deduplicates_responses(in_memory_db, sample_judgement, tmp_path):
    """Test that identical responses are deduplicated."""
    writer = SqlJudgementWriter()

    with writer.open(tmp_path) as w:
        # Write same judgement twice
        result1 = w.write_one(sample_judgement)
        result2 = w.write_one(sample_judgement)

        # Both should succeed, creating different calls
        assert result1.item_id != result2.item_id

    summary = writer.get_summary()
    assert summary.judgements_written == 2


@pytest.mark.integration
def test_sql_writer_handles_null_score(in_memory_db, sample_judgement, tmp_path):
    """Test that writer handles judgements with null score (parsing failure)."""
    # Create judgement with null score
    judgement_with_null_score = LLMJudgement(
        judging_sample=sample_judgement.judging_sample,
        llm_request=sample_judgement.llm_request,
        llm_response=sample_judgement.llm_response,
        llm_score=None,  # Parsing failed
        run_info=sample_judgement.run_info,
    )

    writer = SqlJudgementWriter()

    with writer.open(tmp_path) as w:
        result = w.write_one(judgement_with_null_score)
        assert result.item_type == "llm_call"
        assert result.item_id is not None

    summary = writer.get_summary()
    assert summary.judgements_written == 1
