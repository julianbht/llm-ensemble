# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LLM Ensemble** is a CLI-first research system for evaluating LLM-as-judge ensembles on information retrieval tasks. The project follows a 4-stage pipeline architecture with shared libraries. I'm trying to develop an LLM relevance judging system using Python and Ollama / Hugginface Inference Endpoints, for my bachelor thesis. 
It should be able to read in an easily exchangeable dataset. It should also be able to switch between models. 
Specifically, I am trying to build an LLM-Ensemble which produces relevance judgements for the LLM Judge Challenge dataset by Rahmani et. al. 
I am trying to use as many diverse small models as possible and aggregate their judgement with some function which decides on the final judgement.
Keep in mind that the system will later need to be fully dockerized. Keep in mind 12-factor app design.

### Four Core CLIs

1. **ingest** — Normalize raw IR datasets into `JudgingExample` records (NDJSON/Parquet)
2. **infer** — Run multiple LLM judges over samples, writing per-model judgements
3. **aggregate** — Combine judgements using ensemble strategies (e.g., weighted majority vote)
4. **evaluate** — Compute metrics and generate HTML reports with reproducibility footers

All artifacts are written to `artifacts/runs/<run_id>/` with a manifest, parquet files, and HTML reports.

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
pip install -e .

# Install development dependencies (pytest, coverage tools)
pip install -e ".[dev]"
```

### Running Individual CLIs

```bash
# Ingest
ingest --dataset llm-judge --data-dir ./data --out ./out/samples.ndjson --limit 100

# Alternative: run via python module
python -m llm_ensemble.ingest_cli --help
```

**Note:** It is planned to add a makefile later for convenience. 

### Testing

The project uses pytest for testing. Tests are organized by CLI module (e.g., `src/llm_ensemble/ingest/tests/`).

```bash
# Run all tests
pytest

# Run tests for a specific CLI module
pytest src/llm_ensemble/ingest/tests/

# Run a specific test file
pytest src/llm_ensemble/ingest/tests/test_llm_judge_ingest.py

# Run a specific test class or function
pytest src/llm_ensemble/ingest/tests/test_ingest_cli.py::TestIngestCLI::test_basic_ingest_to_stdout

# Run with verbose output
pytest -v

# Show print statements (useful for debugging)
pytest -v -s

# Run with coverage reporting (requires pytest-cov)
pip install pytest-cov
pytest --cov=llm_ensemble
```

**Test Structure:**
- **Domain/Adapter tests** — Test pure logic and I/O adapters in isolation (e.g., `test_llm_judge_ingest.py`)
- **CLI integration tests** — Test end-to-end CLI behavior (e.g., `test_ingest_cli.py`)

**Note:** pytest is configured in `pyproject.toml` with `-q` (quiet mode) by default.

## Data Contracts

### Canonical Dataset Record (JudgingExample)
- `dataset`: Dataset identifier (e.g., "llm-judge-2024")
- `query_id`, `query_text`: The information need
- `docid`, `doc`: Candidate document to judge
- `gold_relevance`: Ground truth label (for calibration/eval splits)

### Canonical Model Judgement
- `model_id`, `provider`, `version`: Model identity
- `label`: Predicted relevance (`{relevant, partially, irrelevant}` — configurable)
- `score`: Normalized [0,1] confidence
- `confidence`: Self-reported or derived uncertainty
- `rationale`, `raw_text`: Explainability
- `latency_ms`, `retries`, `cost_estimate`: Observability

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
name: llm_judge_challenge
splits:
  train: "data/llm_judge_challenge/train.jsonl"
  test: "data/llm_judge_challenge/test.jsonl"
label_space: [relevant, partially, irrelevant]
```

### Ensembles (`configs/ensembles/*.yaml`)
```yaml
strategy: weighted_majority
params:
  default_weight: 1.0
  per_model_weights:
    phi3-mini: 1.0
    tinyllama: 1.0
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
│   └── tests/       # Unit tests
├── infer/           # Infer logic
├── aggregate/       # Aggregate logic
├── evaluate/        # Evaluate logic
└── libs/            # Shared utilities
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

Reports live at `runs/{run_id}/report.html` for thesis appendices.

## Important Notes



- **12-factor friendly:** CLIs read from files, write to stdout by default, and are configurable via flags/env
- **Environment:** Use `LOG_LEVEL` env var for verbosity
- **No hidden state:** Everything is persisted to disk with manifests
- The `Makefile` paths currently reference `apps/` (old structure) — use direct Python module invocation until updated
- Shared libs in `src/llm_ensemble/libs/` avoid duplication across the four CLIs
