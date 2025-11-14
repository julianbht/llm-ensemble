#!/usr/bin/env python3
"""Initialize database schema by creating all tables.

This script creates all database tables from SQLAlchemy ORM models.
Run once after starting the database (make db) to set up the schema.

Usage:
    python scripts/init_db.py
    # or via Makefile:
    make db-init

The script imports all ORM models so SQLAlchemy knows about them,
then calls create_all_tables() to generate CREATE TABLE statements.

Tables persist in PostgreSQL (stored in Docker volume), so you only
need to run this once unless you:
- Delete the Docker volume (destroys all data)
- Add new ORM models to the codebase
"""

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.db import get_engine, create_schemas, create_all_tables

# Load runtime configuration (reads .env and runtime configs)
load_runtime_config()

# Import all ORM models so they are registered with SQLAlchemy's Base.metadata
# This is required for create_all_tables() to know which tables to create

# Ingest CLI ORMs (datasets, queries, documents, samples)
from llm_ensemble.ingest.schemas.orms import (
    DatasetORM,
    QueryORM,
    DocumentORM,
    IngestRunORM,
    JudgingSampleORM,
)

# Infer CLI ORMs (providers, models, prompts, requests, responses, calls)
from llm_ensemble.infer.schemas.orms_normalized import (
    ProviderORM,
    PromptTemplateORM,
    ModelSpecORM,
    InferRunORM,
    ParserSpecORM,
    LLMRequestORM,
    LLMCallORM,
    LLMResponseORM,
)


def main():
    """Create all database schemas and tables."""
    print("Initializing database schema...")
    print("Reading DATABASE_URL from environment...")

    # Get database engine (reads DATABASE_URL from .env)
    engine = get_engine()

    print(f"Connected to: {engine.url}")
    print("Creating PostgreSQL schemas...")

    # Create schemas first (ingest, infer, aggregate, evaluate)
    create_schemas(engine)

    print("Creating tables...")

    # Create all tables (idempotent - safe to run multiple times)
    create_all_tables(engine)

    print("Database schema initialized successfully!")
    print("")
    print("Schemas and tables created:")
    print("  ingest: datasets, queries, documents, ingest_runs, judging_samples")
    print("  infer: providers, prompt_templates, model_specs, parser_specs,")
    print("         infer_runs, llm_requests, llm_calls, llm_responses")


if __name__ == "__main__":
    main()
