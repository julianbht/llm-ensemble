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

from collections import defaultdict
from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.db import get_engine, create_schemas, create_all_tables, Base

# Load runtime configuration (reads .env and runtime configs)
load_runtime_config()

# Import all ORM models so they are registered with SQLAlchemy's Base.metadata
# This is required for create_all_tables() to know which tables to create

# Ingest CLI ORMs
from llm_ensemble.ingest.schemas.orms import (
    QueryORM,
    DocumentORM,
    NormalizedDatasetORM,
    DatasetSampleORM,
    IngestRunORM,
    JudgingSampleORM,
)

# Infer CLI ORMs
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    ProviderORM,
    ParserORM,
    PromptBuilderORM,
    PromptTemplateORM,
    IngestRunContextORM,
    ModelConfigORM,
    InferRunConfigORM,
    InferRunOutputORM,
    InferRunORM,
    LLMPromptTextORM,
    LLMResponseTextORM,
    LLMScoreORM,
    LLMJudgementORM,
)

# Aggregate CLI ORMs
# from llm_ensemble.aggregate.schemas.orms_normalized import (
#     AggregationSpecORM,
#     AggregateRunORM,
#     AggregatedDatasetORM,
#     AggregatedVoteORM,
#     AggregatedDatasetVoteORM,
#     AggregationVoteORM,
# )


def main():
    """Create all database schemas and tables."""
    print("=" * 70)
    print("DATABASE INITIALIZATION")
    print("=" * 70)
    print()

    # Get database engine (reads DATABASE_URL from .env)
    print("Connecting to database...")
    engine = get_engine()
    print(f"  Connected to: {engine.url.database}@{engine.url.host}:{engine.url.port}")
    print()

    # Create schemas first (ingest, infer, aggregate, evaluate)
    print("Creating PostgreSQL schemas...")
    create_schemas(engine)
    print("  Schemas: public, ingest, infer, aggregate, evaluate")
    print()

    # Create all tables (idempotent - safe to run multiple times)
    print("Creating tables from ORM models...")
    create_all_tables(engine)

    # Dynamically list all created tables grouped by schema
    tables_by_schema = defaultdict(list)
    for table in Base.metadata.sorted_tables:
        schema = table.schema or 'public'
        tables_by_schema[schema].append(table.name)

    total_tables = sum(len(tables) for tables in tables_by_schema.values())
    print(f"  Created {total_tables} tables across {len(tables_by_schema)} schemas")
    print()

    print("=" * 70)
    print("SCHEMA SUMMARY")
    print("=" * 70)
    print()
    for schema in sorted(tables_by_schema.keys()):
        tables = sorted(tables_by_schema[schema])
        print(f"  {schema.upper()} ({len(tables)} tables):")
        for table in tables:
            print(f"    - {table}")
        print()

    print("=" * 70)
    print("DATABASE INITIALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
