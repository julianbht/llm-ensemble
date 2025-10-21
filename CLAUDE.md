# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Ensemble** is a CLI-first LLM relevance labeling system for information retrieval tasks. 
The project follows a 4-stage pipeline architecture with shared libraries.
Specifically, its an LLM relevance judging system using Python with OpenRouter / Ollama / Hugginface Inference Endpoints, for my bachelor thesis. 
It should be able to easily exchange dataset, model and prompt. 


### Four Core CLIs

1. **ingest** — Normalize raw IR datasets into `JudgingExample` records (NDJSON/Parquet)
2. **infer** — Run multiple LLM judges over samples, writing per-model judgements
3. **aggregate** — Combine judgements using ensemble strategies (e.g., weighted majority vote)
4. **evaluate** — Compute metrics and generate HTML reports with reproducibility footers

All artifacts are written to `artifacts/runs/<cli_name>/<run_id>/` with manifests tracking git SHA, timestamps, and full reproducibility metadata.

## Architecture: Clean Architecture / Ports & Adapters

The codebase separates **domain logic** from **infrastructure**:

- **Domain layer** (`domain/`) — Pure Python logic with no I/O. Works with data structures (dicts, DataFrames, Pydantic models). Easy to test and reason about.
  - Example: `ingest/domain/models.py` defines `Query`, `Document`, `Relevance`, and `JudgingExample` schemas

- **Adapters layer** (`adapters/`) — Handles I/O, APIs, file formats, retries, HTTP clients
  - Example: `ingest/adapters/llm_judge.py` reads TSV/JSONL files and yields domain models

- **CLI layer** (`cli/`) — Typer-based entrypoints that wire adapters to domain logic

**Benefits:** Test logic without APIs/GPUs, swap providers easily, refactor layers independently.

## Development Commands

### Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install (use Makefile for convenience)
make install-dev

# Or manually:
pip install -e ".[dev]"
```

### Running Individual CLIs

```bash
# Ingest - Normalize raw datasets into JudgingExamples
ingest --dataset llm_judge_challenge --limit 100
# Uses config file: configs/datasets/llm_judge_challenge.yaml
# Output: artifacts/runs/ingest/<run_id>/samples.ndjson

# Override data directory if needed
ingest --dataset llm_judge_challenge --data-dir /custom/path --limit 100

# Infer - Run LLM judge inference
infer --model gpt-oss-20b --input artifacts/runs/ingest/<run_id>/samples.ndjson
# Output: artifacts/runs/infer/<run_id>/judgements.ndjson

# Alternative: run via python module
python -m llm_ensemble.ingest_cli --help
python -m llm_ensemble.infer_cli --help
```

### Testing

The project uses pytest for testing. Tests are organized in the `tests/` directory, mirroring the CLI structure.

```bash
# Using Makefile (recommended)
make test              # Run all tests
make test-ingest       # Run ingest tests only
make test-infer        # Run infer tests only
make test-schema       # Run schema validation tests only

# Using pytest directly
pytest                 # Run all tests
pytest tests/ingest/   # Run specific module
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

## Data Contracts

### Canonical Dataset Record (JudgingExample)
- `dataset`: Dataset identifier (e.g., "llm-judge-2024")
- `query_id`, `query_text`: The information needv
- `docid`, `doc`: Candidate document to judge
- `gold_relevance`: Ground truth label (for calibration/eval splits)

### Canonical Model Judgement
- `model_id`, `provider`, `version`: Model identity
- `label`: Predicted relevance (0, 1, 2, or null if parsing failed)
- `score`: Relevance score [0-2 scale]
- `confidence`: Model self-reported confidence (optional)
- `rationale`, `raw_text`: Explainability
- `latency_ms`, `retries`, `cost_estimate`: Observability
- `warnings`: Parser warnings, fallbacks

### Ensemble Output
- `final_label`, `final_confidence`: Aggregated decision
- `per_model_votes`: List of individual judgements
- `aggregation_strategy`: Name + params (e.g., "weighted_majority")
- `warnings`: Ties, low agreement, parser fallbacks

## Configuration

### Models (`configs/models/*.yaml`)
```yaml
model_id: tinyllama
provider: ollama
context_window: 2048
default_params:
  temperature: 0.0
  max_tokens: 256
```

### Datasets (`configs/datasets/*.yaml`)
```yaml
dataset_id: llm-judge-2024
adapter: llm_judge  # Maps to ingest/adapters/llm_judge.py
description: "LLM Judge Challenge 2024 - IR relevance judging dataset"
version: "0.1"

# Default data directory (can be overridden via CLI --data-dir)
data_dir: data/llm_judge_challenge

# File paths relative to data_dir
files:
  queries: llm4eval_query_2024.txt
  documents: llm4eval_document_2024.jsonl
  qrels: llm4eval_test_qrel_2024.txt
```

**Note:** The `adapter` field specifies which ingest adapter module to use, making the dataset-adapter coupling explicit.

### Ensembles (`configs/ensembles/*.yaml`)
```yaml
strategy: weighted_majority
params:
  default_weight: 1.0
  per_model_weights:
    phi3-mini: 1.0
    tinyllama: 1.0
```

### Prompts (`configs/prompts/*.yaml`)
```yaml
name: thomas-et-al-prompt
template_file: thomas-et-al-prompt.jinja
description: |
  Thomas et al. search quality rater prompt with structured JSON output.
  Supports optional role description, query context, and multi-aspect evaluation.

# Variables passed to the Jinja2 template
variables:
  role: true           # Include "You are a search quality rater" role description
  aspects: false       # Use simple O-only format instead of M/T/O aspects
  description: null    # Optional query description
  narrative: null      # Optional query narrative

# Output parsing configuration
expected_output_format: json
response_parser: parse_thomas_response
```

## Project Structure

```
src/llm_ensemble/
├── ingest_cli.py    # CLI 1: Dataset normalization entrypoint
├── infer_cli.py     # CLI 2: Model inference entrypoint
├── aggregate_cli.py # CLI 3: Ensemble aggregation entrypoint
├── evaluate_cli.py  # CLI 4: Metrics & reports entrypoint
├── ingest/          # Ingest logic
│   ├── domain/      # Pure logic: models, validation
│   ├── adapters/    # I/O: TSV/JSONL/HF loaders
├── infer/           # Infer logic
├── aggregate/       # Aggregate logic
├── evaluate/        # Evaluate logic
└── libs/            # Shared utilities
    ├── config/      # Config loaders (dataset, model, etc.)
    ├── io/          # Parquet readers/writers
    ├── logging/     # JSON logger
    ├── runtime/     # Environment config
    └── utils/       # Chunking, etc.
```

## Reproducibility Requirements

All reports and artifacts must include:
- **Git SHA** — Exact code version
- **run_id** — Unique run identifier
- **Dataset checksum** — Data integrity
- **Model registry snapshot path** — Model versions used

Reports live at `artifacts/runs/evaluate/{run_id}/report.html` for thesis appendices.

## Important Notes

- **12-factor friendly:** CLIs read from files, write to `artifacts/runs/`, configurable via flags/env
- **Environment variables:** For secrets (API keys) and infrastructure (endpoints)
- **CLI flags:** All task parameters (model, input, dataset) are explicit via required flags
- **Config files:** Dataset configs define adapter mappings and default paths (can be overridden)
- **No hidden state:** Everything persisted to disk with manifests tracking git SHA and full metadata
- **Run management:** All outputs organized by CLI under `artifacts/runs/{ingest,infer,aggregate,evaluate}/`
- Shared libs in `src/llm_ensemble/libs/` avoid duplication across the four CLIs
- Keep in mind that the system will later need to be fully dockerized. 
- Keep in mind 12-factor app design. 
- Follow common software design principles, such as seperation of concerns, to produce clean reusable code.
