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

All artifacts are managed via the **run manager** (`libs/runtime/run_manager.py`), which handles run directory creation, ID generation, and manifest writing. Artifacts are written to `artifacts/runs/<cli_name>/<run_id>/` with manifests tracking git SHA, timestamps, and full reproducibility metadata.

## Architecture: Clean Architecture / Ports & Adapters

The codebase follows hexagonal architecture with clear separation of concerns. Using the **infer** CLI as the reference implementation:

### Layers

1. **CLI Layer** (e.g., `infer_cli.py`)
   - Typer-based entrypoints that parse arguments and delegate to orchestrators
   - No business logic - pure wiring

2. **Orchestrator Layer** (e.g., `infer/orchestrator.py`)
   - Infrastructure concerns: run management, logging, manifests
   - Loads configurations, instantiates adapters via factories
   - Delegates business logic to domain services

3. **Domain Layer** (e.g., `infer/domain/inference_service.py`)
   - Pure business logic with no I/O dependencies
   - Depends only on port abstractions (ABCs)
   - Coordinates the core workflow (read → process → write)

4. **Ports Layer** (e.g., `infer/ports/`)
   - Abstract base classes defining contracts for infrastructure
   - Examples: `LLMProvider`, `ExampleReader`, `JudgementWriter`, `PromptBuilder`, `ResponseParser`

5. **Adapters Layer** (e.g., `infer/adapters/`)
   - Concrete implementations of ports
   - Handle I/O, APIs, file formats, retries, HTTP clients
   - Organized by concern: `io/`, `providers/`, `prompts/`, `parsers/`
   - Instantiated via factory functions (e.g., `get_provider()`, `get_example_reader()`)

### Example: Infer CLI Flow

```
CLI (infer_cli.py)
  ↓
Orchestrator (orchestrator.py) - loads configs, creates run dir, sets up logging
  ↓
Domain Service (InferenceService) - coordinates: reader.read() → provider.infer() → writer.write()
  ↓
Adapters - concrete implementations:
  • NdjsonExampleReader (io/)
  • OpenRouterAdapter (providers/)
  • JinjaPromptBuilder (prompts/)
  • JsonResponseParser (parsers/)
```

**Benefits:** Test domain logic without APIs/GPUs, swap providers via config, refactor layers independently.

## Design Principles

### Explicit Configuration Over Implicit Defaults

**Minimize defaults to ensure users are aware of all behavior.**

- **All adapters** must be explicitly specified via configuration files (no hidden fallbacks, "config first")
- **All CLI behavior** should be visible through flags or configs
- **Configuration files bundle related concerns** (e.g., prompts bundle builder + parser, I/O configs bundle reader + writer)
- **Errors over silent fallbacks** - if config is missing or invalid, raise clear errors explaining what's needed
- **Verbosity confronts users with choices** - this helps them understand how the system works and what they can adjust

**Examples:**
- ✅ `--model gpt-oss-20b` (loads `configs/models/gpt-oss-20b.yaml` which explicitly specifies `provider: openrouter`)
- ✅ `--io ndjson` (loads `configs/io/ndjson.yaml` which explicitly specifies reader and writer adapters)
- ✅ Missing `provider` field in model config → **ValidationError** (not silent default)
- ❌ Don't silently default to a specific provider or I/O format without user knowledge

**Rationale:** Explicit configuration makes the system's behavior transparent and predictable. Users understand what's happening and can adjust behavior by modifying configs, not by discovering hidden defaults through trial and error.

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
# Uses config file: configs/datasets/llm_judge_challenge.yaml
ingest --dataset llm_judge_challenge --limit 100

# Override data directory if needed
ingest --dataset llm_judge_challenge --data-dir /custom/path --limit 100

# Infer - Run LLM judge inference
# Uses configs: configs/models/gpt-oss-20b.yaml, configs/prompts/thomas-et-al-prompt.yaml, configs/io/ndjson.yaml
infer --model gpt-oss-20b --input artifacts/runs/ingest/<run_id>/samples.ndjson

# Explicit prompt and I/O format
infer --model gpt-oss-20b --input data.ndjson --prompt thomas-et-al-prompt --io ndjson

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

The pipeline uses Pydantic models to enforce schemas at CLI boundaries, ensuring type safety and validation across the entire workflow.

### JudgingExample (ingest → infer)

Normalized query-document pairs with ground truth labels. Created by `ingest` CLI, consumed by `infer` CLI.

**Schema:** `ingest/schemas/judging_example.py` → `JudgingExample`
**Key fields:** `dataset`, `query_id`, `query_text`, `docid`, `doc`, `gold_relevance`
**Purpose:** Standardize diverse IR datasets into a single format for downstream processing

### ModelJudgement (infer → aggregate)

LLM-generated relevance assessments with observability metadata. Created by `infer` CLI, consumed by `aggregate` CLI.

**Schema:** `infer/schemas/model_judgement_schema.py` → `ModelJudgement`
**Key fields:**
- **Identity:** `model_id`, `provider`, `version`
- **Judgement:** `label` (0/1/2 or null), `score`, `confidence`
- **Explainability:** `rationale`, `raw_text`
- **Observability:** `latency_ms`, `retries`, `cost_estimate`, `warnings`

**Purpose:** Track both judgements and metadata for analysis, debugging, and cost estimation

### EnsembleOutput (aggregate → evaluate)

Aggregated decisions from multiple models with voting metadata.

**Schema:** TBD in `aggregate/schemas/` (planned)
**Expected fields:** `final_label`, `final_confidence`, `per_model_votes`, `aggregation_strategy`, `warnings`
**Purpose:** Combine multiple model judgements into consensus decisions

## Configuration

All system behavior is controlled via YAML configuration files in `configs/`. CLI flags reference config names (not paths), promoting a "config-first" design.

**Configuration Types:**

- **Models** (`configs/models/*.yaml`) — Provider, context window, default parameters
  - Example: `--model gpt-oss-20b` loads `configs/models/gpt-oss-20b.yaml`
  - Schema: `infer/schemas/model_config_schema.py`

- **Datasets** (`configs/datasets/*.yaml`) — Adapter mapping, file paths, metadata
  - Example: `--dataset llm_judge_challenge` loads `configs/datasets/llm_judge_challenge.yaml`
  - Schema: `libs/config/dataset_loader.py`

- **Prompts** (`configs/prompts/*.yaml`) — Template file, variables, bundled builder + parser
  - Example: `--prompt thomas-et-al-prompt` loads `configs/prompts/thomas-et-al-prompt.yaml`
  - Schema: `infer/schemas/prompt_config_schema.py`

- **I/O Formats** (`configs/io/*.yaml`) — Bundled reader + writer adapters
  - Example: `--io ndjson` loads `configs/io/ndjson.yaml`
  - Schema: `infer/schemas/io_config_schema.py`

- **Ensembles** (`configs/ensembles/*.yaml`) — Strategy, model weights
  - Example: `--ensemble weighted_majority` loads `configs/ensembles/weighted_majority.yaml` (planned)

**Config Overrides:** All configs support runtime overrides via `--override key=value` for experimentation without modifying files.

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
│   ├── domain/      # Pure logic: inference service
│   ├── ports/       # Abstract interfaces (ABCs)
│   ├── adapters/    # Concrete implementations
│   │   ├── io/      # NDJSON reader/writer
│   │   └── providers/ # OpenRouter, Ollama, HF
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
