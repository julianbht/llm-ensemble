#!/usr/bin/env python3
"""Export OFFICIAL ingest and infer runs from database to postgres-friendly SQL dump.

This script:
1. Connects to the source postgres database
2. Queries all OFFICIAL runs (ingest + infer)
3. Exports all related data as INSERT statements
4. Outputs compressed SQL dump for transfer to another postgres server

The export includes:
- All ingest run data (runs, configs, datasets, samples, queries, documents)
- All infer run data (runs, configs, judgements, prompts, responses, scores)
- Preserves all foreign key relationships
- Uses efficient bulk export format

Usage:
    python scripts/export_official_runs_to_postgres.py --output official_runs.sql.gz

    # Transfer to another server
    gunzip -c official_runs.sql.gz | psql $TARGET_DATABASE_URL
"""

import argparse
import gzip
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from llm_ensemble.libs.db.session import get_session


def write_header(f: TextIO) -> None:
    """Write SQL file header with metadata."""
    f.write("-- LLM Ensemble OFFICIAL Runs Export\n")
    f.write(f"-- Generated: {datetime.utcnow().isoformat()}Z\n")
    f.write("-- Contains: OFFICIAL ingest and infer runs with all related data\n")
    f.write("--\n")
    f.write("-- Usage: psql $DATABASE_URL < official_runs.sql\n")
    f.write("--\n\n")
    f.write("BEGIN;\n\n")
    f.write("SET CONSTRAINTS ALL DEFERRED;\n\n")


def write_footer(f: TextIO) -> None:
    """Write SQL file footer."""
    f.write("\nCOMMIT;\n")


def export_ingest_runs(session: Session, f: TextIO) -> None:
    """Export all OFFICIAL ingest runs and related data."""
    f.write("-- ============================================================\n")
    f.write("-- INGEST SCHEMA: OFFICIAL runs\n")
    f.write("-- ============================================================\n\n")

    # Get OFFICIAL ingest runs
    result = session.execute(text("""
        SELECT id FROM ingest.ingest_runs
        WHERE run_type = 'OFFICIAL'
        ORDER BY start_time
    """))
    ingest_run_ids = [str(row[0]) for row in result]

    if not ingest_run_ids:
        f.write("-- No OFFICIAL ingest runs found\n\n")
        return

    f.write(f"-- Found {len(ingest_run_ids)} OFFICIAL ingest runs\n\n")

    # Export queries (referenced by judging samples)
    f.write("-- Queries\n")
    result = session.execute(text("""
        SELECT DISTINCT q.*
        FROM ingest.queries q
        JOIN ingest.judging_samples js ON js.query_id = q.id
        JOIN ingest.normalized_dataset_judging_sample ndjs ON ndjs.judging_sample_id = js.id
        JOIN ingest.normalized_datasets nd ON nd.id = ndjs.normalized_dataset_id
        JOIN ingest.ingest_runs ir ON ir.normalized_dataset_id = nd.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO ingest.queries (id, content_hash, query_text, created_at) VALUES ('{row.id}', '{row.content_hash}', {_quote(row.query_text)}, '{row.created_at}') ON CONFLICT (content_hash) DO NOTHING;\n")
    f.write("\n")

    # Export documents
    f.write("-- Documents\n")
    result = session.execute(text("""
        SELECT DISTINCT d.*
        FROM ingest.documents d
        JOIN ingest.judging_samples js ON js.document_id = d.id
        JOIN ingest.normalized_dataset_judging_sample ndjs ON ndjs.judging_sample_id = js.id
        JOIN ingest.normalized_datasets nd ON nd.id = ndjs.normalized_dataset_id
        JOIN ingest.ingest_runs ir ON ir.normalized_dataset_id = nd.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO ingest.documents (id, content_hash, doc_text, created_at) VALUES ('{row.id}', '{row.content_hash}', {_quote(row.doc_text)}, '{row.created_at}') ON CONFLICT (content_hash) DO NOTHING;\n")
    f.write("\n")

    # Export judging samples
    f.write("-- Judging Samples\n")
    result = session.execute(text("""
        SELECT DISTINCT js.*
        FROM ingest.judging_samples js
        JOIN ingest.normalized_dataset_judging_sample ndjs ON ndjs.judging_sample_id = js.id
        JOIN ingest.normalized_datasets nd ON nd.id = ndjs.normalized_dataset_id
        JOIN ingest.ingest_runs ir ON ir.normalized_dataset_id = nd.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO ingest.judging_samples (id, query_id, document_id, gold_score, created_at) VALUES ('{row.id}', '{row.query_id}', '{row.document_id}', '{row.gold_score}', '{row.created_at}') ON CONFLICT (query_id, document_id) DO NOTHING;\n")
    f.write("\n")

    # Export normalized datasets
    f.write("-- Normalized Datasets\n")
    result = session.execute(text("""
        SELECT DISTINCT nd.*
        FROM ingest.normalized_datasets nd
        JOIN ingest.ingest_runs ir ON ir.normalized_dataset_id = nd.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        ext_name = _quote(row.external_dataset_name) if row.external_dataset_name else 'NULL'
        f.write(f"INSERT INTO ingest.normalized_datasets (id, fingerprint, external_dataset_name, created_at) VALUES ('{row.id}', '{row.fingerprint}', {ext_name}, '{row.created_at}') ON CONFLICT (fingerprint) DO NOTHING;\n")
    f.write("\n")

    # Export normalized dataset judging sample associations
    f.write("-- Normalized Dataset Judging Sample Associations\n")
    result = session.execute(text("""
        SELECT DISTINCT ndjs.*
        FROM ingest.normalized_dataset_judging_sample ndjs
        JOIN ingest.normalized_datasets nd ON nd.id = ndjs.normalized_dataset_id
        JOIN ingest.ingest_runs ir ON ir.normalized_dataset_id = nd.id
        WHERE ir.run_type = 'OFFICIAL'
        ORDER BY ndjs.normalized_dataset_id, ndjs.sequence_number
    """))
    for row in result:
        f.write(f"INSERT INTO ingest.normalized_dataset_judging_sample (id, normalized_dataset_id, judging_sample_id, sequence_number, created_at) VALUES ('{row.id}', '{row.normalized_dataset_id}', '{row.judging_sample_id}', {row.sequence_number}, '{row.created_at}') ON CONFLICT (normalized_dataset_id, judging_sample_id) DO NOTHING;\n")
    f.write("\n")

    # Export ingest run configs
    f.write("-- Ingest Run Configs\n")
    result = session.execute(text("""
        SELECT DISTINCT irc.*
        FROM ingest.ingest_run_configs irc
        JOIN ingest.ingest_runs ir ON ir.ingest_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        limit_val = row.limit if row.limit is not None else 'NULL'
        f.write(f"INSERT INTO ingest.ingest_run_configs (id, io_config_name, input_path, limit, created_at) VALUES ('{row.id}', '{row.io_config_name}', '{row.input_path}', {limit_val}, '{row.created_at}') ON CONFLICT (io_config_name, input_path, limit) DO NOTHING;\n")
    f.write("\n")

    # Export ingest runs
    f.write("-- Ingest Runs\n")
    result = session.execute(text("""
        SELECT * FROM ingest.ingest_runs
        WHERE run_type = 'OFFICIAL'
        ORDER BY start_time
    """))
    for row in result:
        notes_val = _quote(row.notes) if row.notes else 'NULL'
        f.write(f"INSERT INTO ingest.ingest_runs (id, run_name, run_type, ingest_run_config_id, normalized_dataset_id, start_time, end_time, git_sha, git_branch, git_is_dirty, notes, created_at) VALUES ('{row.id}', '{row.run_name}', '{row.run_type}', '{row.ingest_run_config_id}', '{row.normalized_dataset_id}', '{row.start_time}', '{row.end_time}', '{row.git_sha}', '{row.git_branch}', '{row.git_is_dirty}', {notes_val}, '{row.created_at}') ON CONFLICT (run_name) DO NOTHING;\n")
    f.write("\n")


def export_infer_runs(session: Session, f: TextIO) -> None:
    """Export all OFFICIAL infer runs and related data."""
    f.write("-- ============================================================\n")
    f.write("-- INFER SCHEMA: OFFICIAL runs\n")
    f.write("-- ============================================================\n\n")

    # Get OFFICIAL infer runs
    result = session.execute(text("""
        SELECT id FROM infer.infer_runs
        WHERE run_type = 'OFFICIAL'
        ORDER BY start_time
    """))
    infer_run_ids = [str(row[0]) for row in result]

    if not infer_run_ids:
        f.write("-- No OFFICIAL infer runs found\n\n")
        return

    f.write(f"-- Found {len(infer_run_ids)} OFFICIAL infer runs\n\n")

    # Export providers
    f.write("-- Providers\n")
    result = session.execute(text("""
        SELECT DISTINCT p.*
        FROM infer.providers p
        JOIN infer.infer_run_configs irc ON irc.provider_id = p.id
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.providers (id, name, version, created_at) VALUES ('{row.id}', '{row.name}', '{row.version}', '{row.created_at}') ON CONFLICT (name, version) DO NOTHING;\n")
    f.write("\n")

    # Export parsers
    f.write("-- Parsers\n")
    result = session.execute(text("""
        SELECT DISTINCT p.*
        FROM infer.parsers p
        JOIN infer.prompt_templates pt ON pt.parser_id = p.id
        JOIN infer.infer_run_configs irc ON irc.prompt_template_id = pt.id
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.parsers (id, name, version, created_at) VALUES ('{row.id}', '{row.name}', '{row.version}', '{row.created_at}') ON CONFLICT (name, version) DO NOTHING;\n")
    f.write("\n")

    # Export prompt builders
    f.write("-- Prompt Builders\n")
    result = session.execute(text("""
        SELECT DISTINCT pb.*
        FROM infer.prompt_builders pb
        JOIN infer.prompt_templates pt ON pt.prompt_builder_id = pb.id
        JOIN infer.infer_run_configs irc ON irc.prompt_template_id = pt.id
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.prompt_builders (id, name, version, created_at) VALUES ('{row.id}', '{row.name}', '{row.version}', '{row.created_at}') ON CONFLICT (name, version) DO NOTHING;\n")
    f.write("\n")

    # Export prompt templates
    f.write("-- Prompt Templates\n")
    result = session.execute(text("""
        SELECT DISTINCT pt.*
        FROM infer.prompt_templates pt
        JOIN infer.infer_run_configs irc ON irc.prompt_template_id = pt.id
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.prompt_templates (id, name, template_text, prompt_builder_id, parser_id, created_at) VALUES ('{row.id}', '{row.name}', {_quote(row.template_text)}, '{row.prompt_builder_id}', '{row.parser_id}', '{row.created_at}') ON CONFLICT (name) DO NOTHING;\n")
    f.write("\n")

    # Export model configs
    f.write("-- Model Configs\n")
    result = session.execute(text("""
        SELECT DISTINCT mc.*
        FROM infer.model_configs mc
        JOIN infer.infer_run_configs irc ON irc.model_config_id = mc.id
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        context_window = row.context_window if row.context_window is not None else 'NULL'
        capabilities = _quote_json(row.capabilities) if row.capabilities is not None else 'NULL'
        temperature = row.temperature if row.temperature is not None else 'NULL'
        max_tokens = row.max_tokens if row.max_tokens is not None else 'NULL'
        top_p = row.top_p if row.top_p is not None else 'NULL'
        freq_penalty = row.frequency_penalty if row.frequency_penalty is not None else 'NULL'
        pres_penalty = row.presence_penalty if row.presence_penalty is not None else 'NULL'
        seed = row.seed if row.seed is not None else 'NULL'
        additional = _quote_json(row.additional_params) if row.additional_params is not None else 'NULL'

        f.write(f"INSERT INTO infer.model_configs (id, name, name_hint, model_id, context_window, capabilities, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, seed, additional_params, created_at) VALUES ('{row.id}', '{row.name}', '{row.name_hint}', '{row.model_id}', {context_window}, {capabilities}, {temperature}, {max_tokens}, {top_p}, {freq_penalty}, {pres_penalty}, {seed}, {additional}, '{row.created_at}') ON CONFLICT (name) DO NOTHING;\n")
    f.write("\n")

    # Export infer run configs
    f.write("-- Infer Run Configs\n")
    result = session.execute(text("""
        SELECT DISTINCT irc.*
        FROM infer.infer_run_configs irc
        JOIN infer.infer_runs ir ON ir.infer_run_config_id = irc.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        start_idx = row.start_idx if row.start_idx is not None else 'NULL'
        end_idx = row.end_idx if row.end_idx is not None else 'NULL'
        f.write(f"INSERT INTO infer.infer_run_configs (id, model_config_id, provider_id, prompt_template_id, input_run_name, start_idx, end_idx, io_name, created_at) VALUES ('{row.id}', '{row.model_config_id}', '{row.provider_id}', '{row.prompt_template_id}', '{row.input_run_name}', {start_idx}, {end_idx}, '{row.io_name}', '{row.created_at}') ON CONFLICT (model_config_id, provider_id, prompt_template_id, input_run_name, start_idx, end_idx, io_name) DO NOTHING;\n")
    f.write("\n")

    # Export infer run outputs
    f.write("-- Infer Run Outputs\n")
    result = session.execute(text("""
        SELECT DISTINCT iro.*
        FROM infer.infer_run_outputs iro
        JOIN infer.infer_runs ir ON ir.infer_run_output_id = iro.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        fingerprint = f"'{row.sample_fingerprint}'" if row.sample_fingerprint else 'NULL'
        finished = 'true' if row.finished else 'false'
        f.write(f"INSERT INTO infer.infer_run_outputs (id, sample_fingerprint, finished, created_at) VALUES ('{row.id}', {fingerprint}, {finished}, '{row.created_at}') ON CONFLICT DO NOTHING;\n")
    f.write("\n")

    # Export infer runs
    f.write("-- Infer Runs\n")
    result = session.execute(text("""
        SELECT * FROM infer.infer_runs
        WHERE run_type = 'OFFICIAL'
        ORDER BY start_time
    """))
    for row in result:
        output_id = f"'{row.infer_run_output_id}'" if row.infer_run_output_id else 'NULL'
        git_sha = f"'{row.git_sha}'" if row.git_sha else 'NULL'
        git_branch = f"'{row.git_branch}'" if row.git_branch else 'NULL'
        git_dirty = 'true' if row.git_is_dirty else 'false' if row.git_is_dirty is not None else 'NULL'
        notes = _quote(row.notes) if row.notes else 'NULL'
        end_time = f"'{row.end_time}'" if row.end_time else 'NULL'

        f.write(f"INSERT INTO infer.infer_runs (id, run_name, run_type, infer_run_config_id, infer_run_output_id, git_sha, git_branch, git_is_dirty, notes, start_time, end_time, created_at) VALUES ('{row.id}', '{row.run_name}', '{row.run_type}', '{row.infer_run_config_id}', {output_id}, {git_sha}, {git_branch}, {git_dirty}, {notes}, '{row.start_time}', {end_time}, '{row.created_at}') ON CONFLICT (run_name) DO NOTHING;\n")
    f.write("\n")

    # Export LLM prompt texts (deduplicated)
    f.write("-- LLM Prompt Texts\n")
    result = session.execute(text("""
        SELECT DISTINCT lpt.*
        FROM infer.llm_prompt_texts lpt
        JOIN infer.llm_judgements lj ON lj.llm_prompt_text_id = lpt.id
        JOIN infer.infer_run_outputs iro ON iro.id = lj.infer_run_output_id
        JOIN infer.infer_runs ir ON ir.infer_run_output_id = iro.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.llm_prompt_texts (id, content_hash, prompt_text, created_at) VALUES ('{row.id}', '{row.content_hash}', {_quote(row.prompt_text)}, '{row.created_at}') ON CONFLICT (content_hash) DO NOTHING;\n")
    f.write("\n")

    # Export LLM response texts (deduplicated)
    f.write("-- LLM Response Texts\n")
    result = session.execute(text("""
        SELECT DISTINCT lrt.*
        FROM infer.llm_response_texts lrt
        JOIN infer.llm_judgements lj ON lj.llm_response_text_id = lrt.id
        JOIN infer.infer_run_outputs iro ON iro.id = lj.infer_run_output_id
        JOIN infer.infer_runs ir ON ir.infer_run_output_id = iro.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        f.write(f"INSERT INTO infer.llm_response_texts (id, content_hash, llm_response_text, created_at) VALUES ('{row.id}', '{row.content_hash}', {_quote(row.llm_response_text)}, '{row.created_at}') ON CONFLICT (content_hash) DO NOTHING;\n")
    f.write("\n")

    # Export LLM scores (deduplicated)
    f.write("-- LLM Scores\n")
    result = session.execute(text("""
        SELECT DISTINCT ls.*
        FROM infer.llm_scores ls
        JOIN infer.llm_judgements lj ON lj.llm_score_id = ls.id
        JOIN infer.infer_run_outputs iro ON iro.id = lj.infer_run_output_id
        JOIN infer.infer_runs ir ON ir.infer_run_output_id = iro.id
        WHERE ir.run_type = 'OFFICIAL'
    """))
    for row in result:
        confidence = row.confidence if row.confidence is not None else 'NULL'
        rationale = _quote(row.rationale) if row.rationale else 'NULL'
        f.write(f"INSERT INTO infer.llm_scores (id, label, confidence, rationale, created_at) VALUES ('{row.id}', '{row.label}', {confidence}, {rationale}, '{row.created_at}') ON CONFLICT (label, confidence, rationale) DO NOTHING;\n")
    f.write("\n")

    # Export LLM judgements (fact table - can be large, batch by run)
    f.write("-- LLM Judgements\n")
    for infer_run_id in infer_run_ids:
        result = session.execute(text(f"""
            SELECT lj.*
            FROM infer.llm_judgements lj
            JOIN infer.infer_run_outputs iro ON iro.id = lj.infer_run_output_id
            JOIN infer.infer_runs ir ON ir.infer_run_output_id = iro.id
            WHERE ir.id = '{infer_run_id}'
            ORDER BY lj.created_at
        """))

        count = 0
        for row in result:
            score_id = f"'{row.llm_score_id}'" if row.llm_score_id else 'NULL'
            cost_estimate = row.cost_estimate_usd if row.cost_estimate_usd is not None else 'NULL'
            actual_cost = row.actual_cost_usd if row.actual_cost_usd is not None else 'NULL'
            gen_id = _quote(row.generation_id) if row.generation_id else 'NULL'
            prompt_tokens = row.prompt_tokens if row.prompt_tokens is not None else 'NULL'
            completion_tokens = row.completion_tokens if row.completion_tokens is not None else 'NULL'
            total_tokens = row.total_tokens if row.total_tokens is not None else 'NULL'
            parser_code = _quote(row.parser_issue_code) if row.parser_issue_code else 'NULL'
            parser_msg = _quote(row.parser_issue_message) if row.parser_issue_message else 'NULL'
            parser_meta = _quote_json(row.parser_issue_metadata) if row.parser_issue_metadata else 'NULL'

            f.write(f"INSERT INTO infer.llm_judgements (id, infer_run_output_id, normalized_dataset_judging_sample_id, llm_prompt_text_id, llm_response_text_id, llm_score_id, latency_ms, retries, cost_estimate_usd, actual_cost_usd, generation_id, prompt_tokens, completion_tokens, total_tokens, parser_issue_code, parser_issue_message, parser_issue_metadata, created_at) VALUES ('{row.id}', '{row.infer_run_output_id}', '{row.normalized_dataset_judging_sample_id}', '{row.llm_prompt_text_id}', '{row.llm_response_text_id}', {score_id}, {row.latency_ms}, {row.retries}, {cost_estimate}, {actual_cost}, {gen_id}, {prompt_tokens}, {completion_tokens}, {total_tokens}, {parser_code}, {parser_msg}, {parser_meta}, '{row.created_at}') ON CONFLICT (infer_run_output_id, normalized_dataset_judging_sample_id) DO NOTHING;\n")
            count += 1

        if count > 0:
            f.write(f"-- Exported {count} judgements for run {infer_run_id}\n")

    f.write("\n")


def _quote(text: str | None) -> str:
    """Quote text for SQL, handling None and escaping single quotes."""
    if text is None:
        return 'NULL'
    # Escape single quotes by doubling them, wrap in dollar quotes to avoid issues
    escaped = text.replace("'", "''")
    return f"$${escaped}$$"


def _quote_json(obj: dict | list | None) -> str:
    """Quote JSON object for SQL."""
    if obj is None:
        return 'NULL'
    import json
    json_str = json.dumps(obj)
    return _quote(json_str) + "::jsonb"


def main():
    parser = argparse.ArgumentParser(
        description="Export OFFICIAL ingest and infer runs to postgres-friendly SQL dump"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output file path (use .sql.gz for compressed output)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="Database URL (defaults to DATABASE_URL env var)",
    )

    args = parser.parse_args()

    # Get database URL
    import os
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not set and --database-url not provided", file=sys.stderr)
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
