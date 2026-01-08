#!/usr/bin/env python3
"""Sync OpenRouter models to PostgreSQL database.

This script:
1. Fetches all models from OpenRouter API
2. Extracts parameter size using heuristics
3. Upserts models into a PostgreSQL table for easy querying

Usage:
    python scripts/sync_openrouter_models_to_db.py
    python scripts/sync_openrouter_models_to_db.py --dry-run
    python scripts/sync_openrouter_models_to_db.py --debug
"""

import sys
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import requests
import typer
from sqlalchemy import create_engine, text

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.logging.structlog_logger import get_logger

# Load environment variables
load_runtime_config()

app = typer.Typer(
    add_completion=True,
    help="Sync OpenRouter models to PostgreSQL database",
    pretty_exceptions_enable=False,
)

logger = get_logger(component="sync_openrouter_models")

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# SQL schema for the table
CREATE_TABLE_SQL = """
CREATE SCHEMA IF NOT EXISTS openrouter;

CREATE TABLE IF NOT EXISTS openrouter.models (
    model_id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    prompt_cost_per_1m DECIMAL(20, 10),
    completion_cost_per_1m DECIMAL(20, 10),
    avg_cost_per_1m DECIMAL(20, 10),
    is_free BOOLEAN NOT NULL,
    param_size DECIMAL(10, 2),  -- in billions, NULL if not detected
    context_length INTEGER,
    last_updated TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
"""

UPSERT_MODEL_SQL = """
INSERT INTO openrouter.models (
    model_id,
    model_name,
    prompt_cost_per_1m,
    completion_cost_per_1m,
    avg_cost_per_1m,
    is_free,
    param_size,
    context_length,
    last_updated
) VALUES (
    :model_id,
    :model_name,
    :prompt_cost_per_1m,
    :completion_cost_per_1m,
    :avg_cost_per_1m,
    :is_free,
    :param_size,
    :context_length,
    :last_updated
)
ON CONFLICT (model_id) DO UPDATE SET
    model_name = EXCLUDED.model_name,
    prompt_cost_per_1m = EXCLUDED.prompt_cost_per_1m,
    completion_cost_per_1m = EXCLUDED.completion_cost_per_1m,
    avg_cost_per_1m = EXCLUDED.avg_cost_per_1m,
    is_free = EXCLUDED.is_free,
    param_size = EXCLUDED.param_size,
    context_length = EXCLUDED.context_length,
    last_updated = EXCLUDED.last_updated;
"""


def fetch_openrouter_models(debug: bool = False) -> list[dict]:
    """Fetch models and pricing from OpenRouter API.

    Args:
        debug: Whether to print debug information

    Returns:
        List of model dicts with all extracted information
    """
    logger.info("fetching_models", url=OPENROUTER_MODELS_URL)

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        logger.error("api_request_failed", error=str(e))
        raise typer.Exit(code=1)

    if "data" not in data:
        logger.error("invalid_api_response", error="missing 'data' key")
        raise typer.Exit(code=1)

    models = []
    free_models_count = 0
    models_with_params = 0
    total_models_count = 0

    for model in data["data"]:
        model_id = model.get("id")
        model_name = model.get("name", "")
        pricing = model.get("pricing", {})
        context_length = model.get("context_length")

        if not model_id or not pricing:
            continue

        total_models_count += 1

        # Parse pricing
        try:
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))

            # Scale to per-1M tokens
            prompt_cost_per_1m = prompt_price * 1_000_000
            completion_cost_per_1m = completion_price * 1_000_000
            avg_cost_per_1m = (prompt_cost_per_1m + completion_cost_per_1m) / 2

            is_free = prompt_price == 0 and completion_price == 0
            if is_free:
                free_models_count += 1

            # Extract parameter size using regex heuristic
            # Pattern: (\d+(?:\.\d+)?)[bB] matches "7b", "70B", "1.5b", etc.
            param_match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_id + " " + model_name)
            param_size = None
            if param_match:
                param_size = float(param_match.group(1))
                models_with_params += 1

            models.append({
                "model_id": model_id,
                "model_name": model_name,
                "prompt_cost_per_1m": prompt_cost_per_1m,
                "completion_cost_per_1m": completion_cost_per_1m,
                "avg_cost_per_1m": avg_cost_per_1m,
                "is_free": is_free,
                "param_size": param_size,
                "context_length": context_length,
            })

            if debug and len(models) <= 3:
                logger.info(
                    "model_parsed",
                    model_id=model_id,
                    param_size=param_size,
                    is_free=is_free,
                )

        except (ValueError, TypeError) as e:
            logger.warning("failed_to_parse_model", model_id=model_id, error=str(e))
            continue

    logger.info(
        "models_fetched",
        total=total_models_count,
        free=free_models_count,
        with_param_size=models_with_params,
        without_param_size=total_models_count - models_with_params,
    )

    return models


def create_table_if_not_exists(engine) -> None:
    """Create the openrouter.models table if it doesn't exist."""
    logger.info("creating_table_schema")

    with engine.connect() as conn:
        # Execute each statement separately
        for statement in CREATE_TABLE_SQL.strip().split(';'):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
                conn.commit()

    logger.info("table_schema_ready")


def sync_models_to_db(engine, models: list[dict], dry_run: bool) -> dict[str, int]:
    """Sync models to database.

    Args:
        engine: SQLAlchemy engine
        models: List of model dicts
        dry_run: If True, don't actually write to database

    Returns:
        Stats dict with counts
    """
    if dry_run:
        logger.info("dry_run_mode", message="Would sync models but skipping due to --dry-run")
        return {"total": len(models), "inserted": 0, "updated": 0}

    logger.info("syncing_models_to_db", count=len(models))

    timestamp = datetime.now(timezone.utc)
    inserted = 0
    updated = 0

    with engine.begin() as conn:
        for model in models:
            # Check if model exists
            result = conn.execute(
                text("SELECT model_id FROM openrouter.models WHERE model_id = :model_id"),
                {"model_id": model["model_id"]}
            )
            exists = result.fetchone() is not None

            # Upsert model
            conn.execute(
                text(UPSERT_MODEL_SQL),
                {
                    **model,
                    "last_updated": timestamp,
                }
            )

            if exists:
                updated += 1
            else:
                inserted += 1

    logger.info(
        "sync_complete",
        total=len(models),
        inserted=inserted,
        updated=updated,
    )

    return {
        "total": len(models),
        "inserted": inserted,
        "updated": updated,
    }


@app.command()
def sync(
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Fetch models but don't write to database",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Show debug information",
        ),
    ] = False,
) -> None:
    """Sync OpenRouter models to PostgreSQL database.

    This command:
    1. Creates openrouter.models table if it doesn't exist
    2. Fetches all models from OpenRouter API
    3. Upserts models into the database

    After syncing, you can query models with SQL:

        -- Free models ≤ 8B
        SELECT model_id, param_size FROM openrouter.models
        WHERE is_free = true AND param_size <= 8
        ORDER BY param_size;

        -- Top 10 cheapest paid models
        SELECT model_id, avg_cost_per_1m, param_size
        FROM openrouter.models
        WHERE is_free = false
        ORDER BY avg_cost_per_1m
        LIMIT 10;

    Examples:

        # Sync models to database
        python scripts/sync_openrouter_models_to_db.py

        # Preview what would be synced
        python scripts/sync_openrouter_models_to_db.py --dry-run
    """
    print("Sync OpenRouter Models to Database")
    print("===================================")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n")

    # Get database engine
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL environment variable not set", file=sys.stderr)
        raise typer.Exit(code=1)

    engine = create_engine(database_url)

    # Create table if needed
    if not dry_run:
        create_table_if_not_exists(engine)

    # Fetch models from API
    models = fetch_openrouter_models(debug)

    # Sync to database
    stats = sync_models_to_db(engine, models, dry_run)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Total models: {stats['total']}")
    if not dry_run:
        print(f"  Inserted: {stats['inserted']}")
        print(f"  Updated: {stats['updated']}")
    print(f"\nQuery the models table:")
    print(f"  SELECT * FROM openrouter.models WHERE is_free = true LIMIT 10;")
    if dry_run:
        print(f"\nRun without --dry-run to sync to database")


if __name__ == "__main__":
    app()
