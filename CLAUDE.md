# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Ensemble** is a CLI-first LLM relevance judging system for information retrieval tasks, built with Python for bachelor thesis research. It supports OpenRouter, Ollama, and HuggingFace Inference Endpoints, with easy swapping of datasets, models, and prompts via configuration files.

The project follows a 4-stage pipeline architecture with shared libraries and hexagonal architecture principles.

### Four Core CLIs

1. **ingest** — Normalize raw IR datasets into `JudgingExample` records
2. **infer** — Run LLM judges over samples, writing per-model judgements
3. **aggregate** — Combine judgements using ensemble strategies (weighted majority vote, etc.)
4. **evaluate** — Compute metrics and generate reports

### Running Individual CLIs

```bash
# Ingest - Normalize raw datasets into JudgingExamples
# Uses config file: configs/io/llm_judge_ingest.yaml
ingest --io llm_judge_ingest --limit 100

# Override dataset directory if needed
ingest --io llm_judge_ingest --override datasets_dir=/custom/path --limit 100

# Infer - Run LLM judge inference
infer --model gpt-oss-20b --prompt thomas-et-al-prompt --io json --input artifacts/runs/ingest/<run_name>/samples.json

# Aggregate - Combine model judgements using ensemble strategies
aggregate --ensemble weighted_majority --io json --input artifacts/runs/infer/<run_name>/judgements.json

# Alternative: run via python module
python3 -m llm_ensemble.ingest_cli --help
python3 -m llm_ensemble.infer_cli --help
python3 -m llm_ensemble.aggregate_cli --help
python3 -m llm_ensemble.evaluate_cli --help

## Architecture: Clean Architecture / Ports & Adapters

The codebase follows hexagonal architecture with clear separation of concerns. Using the **infer** CLI as the reference implementation:

## Design Principles

### Explicit Configuration Over Implicit Defaults

**Minimize defaults to ensure users are aware of all behavior.**

- **All adapters** must be explicitly specified via configuration files (no hidden fallbacks, "config first")
- **All CLI behavior** should be visible through flags or configs
- **Configuration files bundle related concerns** (e.g., prompts bundle builder + parser, I/O configs bundle reader + writer)
- **Errors over silent fallbacks** - if config is missing or invalid, raise clear errors explaining what's needed
- **Verbosity confronts users with choices** - this helps them understand how the system works and what they can adjust

**Rationale:** Explicit configuration makes the system's behavior transparent and predictable. Users understand what's happening and can adjust behavior by modifying configs, not by discovering hidden defaults through trial and error.

## Development Commands

### Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install (use Makefile for convenience)
make install-dev

# Or manually:
pip install -e ".[dev]"

```
### Testing

The project uses pytest for testing. Tests are organized in the `tests/` directory, mirroring the CLI structure.

```bash
# Using Makefile (recommended)
make test              # Run all tests
make test-ingest       # Run ingest tests only
make test-infer        # Run infer tests only

# Using pytest directly
pytest                 # Run all tests
pytest tests/ingest/   # Run specific module
pytest tests/infer/test_inference_service.py  # Run single test file
pytest tests/infer/test_inference_service.py::test_inference_pipeline  # Run single test
pytest -v              # Verbose output
pytest -v -s           # Show print statements
pytest --cov=llm_ensemble  # Coverage report

# Using markers
pytest -m unit         # Run only unit tests (fast, isolated)
pytest -m integration  # Run integration tests (file I/O, adapters)
pytest -m "not slow"   # Skip slow tests
pytest -m requires_api # Run tests requiring API credentials
```

**Test Structure:**
- **Domain/Adapter tests** — Test pure logic and I/O adapters in isolation (e.g., `test_llm_judge_ingest.py`)
- **CLI integration tests** — Test end-to-end CLI behavior (e.g., `test_ingest_cli.py`)

**Test Markers:**
- `@pytest.mark.unit` — Fast, isolated tests with no I/O
- `@pytest.mark.integration` — Tests using files or adapters
- `@pytest.mark.slow` — Long-running tests or API calls
- `@pytest.mark.requires_api` — Tests requiring API credentials

**Configuration:** Tests are discovered from `tests/` directory. pytest is configured in `pyproject.toml` with `-q` (quiet mode) by default.

### Environment Variables

The project uses environment variables for sensitive credentials and infrastructure configuration:

```bash
# Create .env file in project root (gitignored)
OPENROUTER_API_KEY=your_openrouter_key      # For OpenRouter models
HF_TOKEN=your_huggingface_token             # For HuggingFace models
OLLAMA_BASE_URL=http://localhost:11434      # For local Ollama (optional)
```

The project uses `python-dotenv` to automatically load `.env` files. Never commit credentials to git.

## Data Contracts

The pipeline uses Pydantic models to enforce schemas at CLI boundaries, ensuring type safety and validation across the entire workflow.

## Important Notes

- **12-factor friendly:** CLIs read from files, write to `artifacts/runs/`, configurable via flags/env
- **Environment variables:** For secrets (API keys) and infrastructure (endpoints)
- **CLI flags:** All task parameters (model, input, I/O format) are explicit via required flags
- **Config files:** I/O configs bundle reader/writer adapters and dataset paths (can be overridden via `--override`)
- **Unified I/O:** All CLIs use the `--io` flag for consistent I/O configuration across the pipeline
- **No hidden state:** Everything persisted to disk with manifests tracking git SHA and full metadata
- **Run management:** All outputs organized by CLI under `artifacts/runs/{ingest,infer,aggregate,evaluate}/`
- Shared libs in `src/llm_ensemble/libs/` avoid duplication across the four CLIs
- Keep in mind that the system will later need to be fully dockerized.
- Keep in mind 12-factor app design.
- Follow common software design principles, such as seperation of concerns, to produce clean reusable code.
- Keep comments and docstrings short and precise. Use expressive names for variables, methods, interfaces etc., even if they may be long.