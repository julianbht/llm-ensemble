#!/usr/bin/env python3
"""Reset database by dropping and recreating all schemas and tables.

This script is useful for presentation demos where you want to start fresh
and re-run the same commands with the same run names.

Usage:
    python scripts/db/reset_db.py
    # or via Makefile:
    make db-reset

WARNING: This will DELETE ALL DATA in the database!
"""

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.db.base import get_engine, create_schemas, create_all_tables, Base
from sqlalchemy import text

# Load runtime configuration (reads .env and runtime configs)
load_runtime_config()

# Import all ORM models so they are registered with SQLAlchemy's Base.metadata
# Ingest CLI ORMs
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    QueryORM,
    DocumentORM,
    NormalizedDatasetORM,
    NormalizedDatasetJudgingSampleORM,
    IngestRunConfigORM,
    IngestRunORM,
    JudgingSampleORM,
)

# Infer CLI ORMs
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    ProviderORM,
    ParserORM,
    PromptBuilderORM,
    PromptTemplateORM,
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
from llm_ensemble.aggregate.adapters.driven.io.orms import (
    AggregationStrategyORM,
    AggregateRunConfigORM,
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
    AggregatedDatasetVoteORM,
    AggregationVoteORM,
)


def main():
    """Drop all schemas and recreate them with tables."""
    print("=" * 70)
    print("DATABASE RESET - WARNING: ALL DATA WILL BE DELETED!")
    print("=" * 70)
    print()

    # Get database engine
    print("Connecting to database...")
    engine = get_engine()
    print(f"  Connected to: {engine.url.database}@{engine.url.host}:{engine.url.port}")
    print()

    # Drop all schemas (CASCADE will drop all tables within them)
    print("Dropping all schemas (CASCADE)...")
    schemas = ["ingest", "infer", "aggregate", "evaluate"]

    with engine.begin() as conn:
        for schema in schemas:
            print(f"  Dropping schema: {schema}")
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))

    print("  All schemas dropped!")
    print()

    # Recreate schemas
    print("Recreating PostgreSQL schemas...")
    create_schemas(engine)
    print("  Schemas: public, ingest, infer, aggregate, evaluate")
    print()

    # Recreate all tables
    print("Recreating tables from ORM models...")
    create_all_tables(engine)

    # Count tables created
    from collections import defaultdict
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
    print("DATABASE RESET COMPLETE - Fresh schemas and tables created!")
    print("=" * 70)


if __name__ == "__main__":
    main()
