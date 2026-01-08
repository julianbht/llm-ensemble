#!/usr/bin/env python3
"""Export OFFICIAL ingest and infer runs using ORM models.

This script uses SQLAlchemy ORM models to export all OFFICIAL runs
and their related data in a postgres-friendly SQL dump format.

Usage:
    python scripts/export_official_runs_orm.py --output official_runs.sql.gz

    # Transfer to another server
    gunzip -c official_runs.sql.gz | psql $TARGET_DATABASE_URL
"""

import argparse
import gzip
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from llm_ensemble.libs.runtime.env import load_runtime_config

# Load runtime configuration (DATABASE_URL, API keys, etc.)
load_runtime_config()

from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    IngestRunORM,
    IngestRunConfigORM,
    NormalizedDatasetORM,
    NormalizedDatasetJudgingSampleORM,
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
)
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunConfigORM,
    InferRunOutputORM,
    ModelConfigORM,
    ProviderORM,
    PromptTemplateORM,
    PromptBuilderORM,
    ParserORM,
    LLMJudgementORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMScoreORM,
)
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.libs.runtime.run_info import RunType


def write_header(f: TextIO) -> None:
    """Write SQL file header."""
    f.write("-- LLM Ensemble OFFICIAL Runs Export\n")
    f.write(f"-- Generated: {datetime.utcnow().isoformat()}Z\n")
    f.write("-- Contains: OFFICIAL ingest and infer runs with all related data\n\n")
    f.write("BEGIN;\n")
    f.write("SET CONSTRAINTS ALL DEFERRED;\n\n")


def write_footer(f: TextIO) -> None:
    """Write SQL file footer."""
    f.write("\nCOMMIT;\n")


def sql_value(val: Any) -> str:
    """Convert Python value to SQL literal."""
    if val is None:
        return 'NULL'
    if isinstance(val, bool):
        return 'true' if val else 'false'
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, (dict, list)):
        return f"$${json.dumps(val)}$$::jsonb"
    # String - escape single quotes
    escaped = str(val).replace("'", "''")
    return f"$${escaped}$$"


def insert_stmt(table: str, obj: Any, conflict_clause: str = "") -> str:
    """Generate INSERT statement from ORM object."""
    columns = []
    values = []

    mapper = obj.__class__.__mapper__
    for col in mapper.columns:
        columns.append(col.name)
        values.append(sql_value(getattr(obj, col.name)))

    cols_str = ", ".join(columns)
    vals_str = ", ".join(values)

    return f"INSERT INTO {table} ({cols_str}) VALUES ({vals_str}){conflict_clause};\n"


def export_ingest_runs(session: Session, f: TextIO) -> None:
    """Export all OFFICIAL ingest runs and related data."""
    f.write("-- INGEST SCHEMA: OFFICIAL runs\n\n")

    # Get all OFFICIAL ingest runs with relationships eagerly loaded
    ingest_runs = session.query(IngestRunORM).filter(
        IngestRunORM.run_type == RunType.OFFICIAL
    ).order_by(IngestRunORM.start_time).all()

    if not ingest_runs:
        f.write("-- No OFFICIAL ingest runs found\n\n")
        return

    f.write(f"-- Found {len(ingest_runs)} OFFICIAL ingest runs\n\n")

    # Track unique entities to avoid duplicates
    queries = set()
    documents = set()
    judging_samples = set()
    normalized_datasets = set()
    dataset_samples = set()
    run_configs = set()

    # Collect all related entities
    for run in ingest_runs:
        run_configs.add(run.ingest_run_config)
        normalized_datasets.add(run.normalized_dataset)

        for ds_sample in run.normalized_dataset.dataset_samples:
            dataset_samples.add(ds_sample)
            judging_samples.add(ds_sample.judging_sample)
            queries.add(ds_sample.judging_sample.query)
            documents.add(ds_sample.judging_sample.document)

    # Export in dependency order
    f.write("-- Queries\n")
    for query in sorted(queries, key=lambda x: x.content_hash):
        f.write(insert_stmt("ingest.queries", query, " ON CONFLICT (content_hash) DO NOTHING"))
    f.write("\n")

    f.write("-- Documents\n")
    for doc in sorted(documents, key=lambda x: x.content_hash):
        f.write(insert_stmt("ingest.documents", doc, " ON CONFLICT (content_hash) DO NOTHING"))
    f.write("\n")

    f.write("-- Judging Samples\n")
    for sample in judging_samples:
        f.write(insert_stmt("ingest.judging_samples", sample, " ON CONFLICT (query_id, document_id) DO NOTHING"))
    f.write("\n")

    f.write("-- Normalized Datasets\n")
    for dataset in normalized_datasets:
        f.write(insert_stmt("ingest.normalized_datasets", dataset, " ON CONFLICT (fingerprint) DO NOTHING"))
    f.write("\n")

    f.write("-- Normalized Dataset Judging Sample Associations\n")
    for ds_sample in sorted(dataset_samples, key=lambda x: (str(x.normalized_dataset_id), x.sequence_number)):
        f.write(insert_stmt("ingest.normalized_dataset_judging_sample", ds_sample, " ON CONFLICT (normalized_dataset_id, judging_sample_id) DO NOTHING"))
    f.write("\n")

    f.write("-- Ingest Run Configs\n")
    for config in run_configs:
        f.write(insert_stmt("ingest.ingest_run_configs", config, " ON CONFLICT (io_config_name, input_path, limit) DO NOTHING"))
    f.write("\n")

    f.write("-- Ingest Runs\n")
    for run in ingest_runs:
        f.write(insert_stmt("ingest.ingest_runs", run, " ON CONFLICT (run_name) DO NOTHING"))
    f.write("\n")


def export_infer_runs(session: Session, f: TextIO) -> None:
    """Export all OFFICIAL infer runs and related data."""
    f.write("-- INFER SCHEMA: OFFICIAL runs\n\n")

    # Get all OFFICIAL infer runs
    infer_runs = session.query(InferRunORM).filter(
        InferRunORM.run_type == RunType.OFFICIAL
    ).order_by(InferRunORM.start_time).all()

    if not infer_runs:
        f.write("-- No OFFICIAL infer runs found\n\n")
        return

    f.write(f"-- Found {len(infer_runs)} OFFICIAL infer runs\n\n")

    # Track unique entities
    providers = set()
    parsers = set()
    prompt_builders = set()
    prompt_templates = set()
    model_configs = set()
    run_configs = set()
    run_outputs = set()
    prompt_texts = set()
    response_texts = set()
    scores = set()
    judgements = []

    # Collect all related entities
    for run in infer_runs:
        run_configs.add(run.infer_run_config)

        config = run.infer_run_config
        providers.add(config.provider)
        model_configs.add(config.model_config)
        prompt_templates.add(config.prompt_template)
        prompt_builders.add(config.prompt_template.prompt_builder)
        parsers.add(config.prompt_template.parser)

        if run.infer_run_output:
            run_outputs.add(run.infer_run_output)

            for judgement in run.infer_run_output.llm_judgements:
                judgements.append(judgement)
                prompt_texts.add(judgement.llm_prompt_text)
                response_texts.add(judgement.llm_response_text)
                if judgement.llm_score:
                    scores.add(judgement.llm_score)

    # Export in dependency order
    f.write("-- Providers\n")
    for provider in providers:
        f.write(insert_stmt("infer.providers", provider, " ON CONFLICT (name, version) DO NOTHING"))
    f.write("\n")

    f.write("-- Parsers\n")
    for parser in parsers:
        f.write(insert_stmt("infer.parsers", parser, " ON CONFLICT (name, version) DO NOTHING"))
    f.write("\n")

    f.write("-- Prompt Builders\n")
    for builder in prompt_builders:
        f.write(insert_stmt("infer.prompt_builders", builder, " ON CONFLICT (name, version) DO NOTHING"))
    f.write("\n")

    f.write("-- Prompt Templates\n")
    for template in prompt_templates:
        f.write(insert_stmt("infer.prompt_templates", template, " ON CONFLICT (name) DO NOTHING"))
    f.write("\n")

    f.write("-- Model Configs\n")
    for model_config in model_configs:
        f.write(insert_stmt("infer.model_configs", model_config, " ON CONFLICT (name) DO NOTHING"))
    f.write("\n")

    f.write("-- Infer Run Configs\n")
    for config in run_configs:
        f.write(insert_stmt("infer.infer_run_configs", config, " ON CONFLICT (model_config_id, provider_id, prompt_template_id, input_run_name, start_idx, end_idx, io_name) DO NOTHING"))
    f.write("\n")

    f.write("-- Infer Run Outputs\n")
    for output in run_outputs:
        f.write(insert_stmt("infer.infer_run_outputs", output, " ON CONFLICT DO NOTHING"))
    f.write("\n")

    f.write("-- Infer Runs\n")
    for run in infer_runs:
        f.write(insert_stmt("infer.infer_runs", run, " ON CONFLICT (run_name) DO NOTHING"))
    f.write("\n")

    f.write("-- LLM Prompt Texts\n")
    for prompt_text in sorted(prompt_texts, key=lambda x: x.content_hash):
        f.write(insert_stmt("infer.llm_prompt_texts", prompt_text, " ON CONFLICT (content_hash) DO NOTHING"))
    f.write("\n")

    f.write("-- LLM Response Texts\n")
    for response_text in sorted(response_texts, key=lambda x: x.content_hash):
        f.write(insert_stmt("infer.llm_response_texts", response_text, " ON CONFLICT (content_hash) DO NOTHING"))
    f.write("\n")

    f.write("-- LLM Scores\n")
    for score in scores:
        f.write(insert_stmt("infer.llm_scores", score, " ON CONFLICT (label, confidence, rationale) DO NOTHING"))
    f.write("\n")

    f.write("-- LLM Judgements\n")
    for judgement in judgements:
        f.write(insert_stmt("infer.llm_judgements", judgement, " ON CONFLICT (infer_run_output_id, normalized_dataset_judging_sample_id) DO NOTHING"))
    f.write(f"-- Exported {len(judgements)} judgements\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export OFFICIAL ingest and infer runs using ORM models"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path (use .sql.gz for compressed output)",
    )

    args = parser.parse_args()

    # Get database URL from environment (loaded via load_runtime_config())
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not set in environment", file=sys.stderr)
        sys.exit(1)

    # Create engine and session
    engine = create_engine(database_url)
    session = get_session(engine)

    # Open output file (compressed if .gz extension)
    output_path = args.output
    if output_path.suffix == ".gz":
        f = gzip.open(output_path, "wt", encoding="utf-8")
    else:
        f = output_path.open("w", encoding="utf-8")

    try:
        print(f"Exporting OFFICIAL runs to {output_path}...")

        write_header(f)
        export_ingest_runs(session, f)
        export_infer_runs(session, f)
        write_footer(f)

        print(f"Export complete: {output_path}")

        # Print file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")

    finally:
        f.close()
        session.close()


if __name__ == "__main__":
    main()
