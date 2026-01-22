# CLAUDE.md

## Project Overview

LLM Ensemble is a framework for creating relevance judgements for information retrieval datasets using an ensemble of LLMs or a single model. It supports evaluating and comparing LLM performance on ranking tasks. 

## Architecture

**Hexagonal Architecture (Ports & Adapters)** - Each module follows this pattern:
- `adapters/driving/` - CLI entry points (Typer apps)
- `adapters/driven/` - External integrations (DB, APIs)
- `application/` - Use cases and orchestration
- `application/ports/` - Interface definitions
- `domain/` - Pure business logic and entities

## Quick Commands

```bash
# Setup
make install-dev    # Install with dev dependencies
make db && make db-init  # Start and initialize PostgreSQL

# Testing
make test           # Run all tests
pytest tests/ingest/  # Run specific module tests

# Infrastructure
make infra          # Start all services (DB + Grafana/Loki)
make infra-down     # Stop all services
```

## CLI Tools

Five main CLIs defined in `pyproject.toml`:
- `ingest` - Load datasets into the system
- `infer` - Run LLM inference on datasets
- `merge-infer-runs` - Combine multiple inference runs
- `aggregate` - Apply ensemble voting strategies (majority, average, random)
- `evaluate` - Compute metrics (Cohen's Kappa, Krippendorff's Alpha)

## Key Directories

- `src/llm_ensemble/` - Main source code
- `configs/models/` - LLM model configurations
- `configs/prompts/` - Prompt templates
- `artifacts/runs/` - Run outputs
- `datasets/` - LLM4Eval challenge data

## Testing

```bash
pytest                    # All tests
pytest -m unit           # Fast, isolated tests
pytest -m integration    # Tests with I/O
```

Test markers defined in `pyproject.toml`: `unit`, `integration`, `slow`, `requires_api`

## Code Conventions

- **Pydantic** for all data validation and config models
- **Typer** for CLI interfaces with type hints
- **SQLAlchemy 2.0** with async patterns for database
- **structlog** for structured JSON logging
- Strict separation between domain logic and infrastructure
