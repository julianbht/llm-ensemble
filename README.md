# LLM Ensemble

A framework for creating relevance judgements for information retrieval datasets using an ensemble of LLMs or a single model.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/yourusername/llm-ensemble.git
cd llm-ensemble
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys (OPENROUTER_API_KEY, etc.)

# 3. Start database
make db
make db-init

# 4. Run your first pipeline
ingest --input-path datasets/llm_judge_challenge_experiment \
       --io-cfg llm_judge_challenge_to_db \
       --run-name my-first-ingest

infer --model-cfg gemma-3n-e2b-it-free \
      --provider openrouter \
      --prompt-template thomas-simple \
      --io-cfg db_to_db \
      --input my-first-ingest

evaluate --io-cfg db_infer_to_json --input-run-name <infer-run-name>
```

## Directory Structure

```
llm-ensemble/
├── src/llm_ensemble/          # Main source code
│   ├── ingest/                # Data ingestion pipeline
│   ├── infer/                 # LLM inference pipeline
│   ├── aggregate/             # Ensemble voting pipeline
│   ├── evaluate/              # Metrics computation pipeline
│   └── libs/                  # Shared utilities (CLI, config, DB, logging)
│
├── configs/
│   ├── models/                # LLM model configurations (20+ models)
│   ├── prompts/               # Prompt templates
│   └── retries/               # Retry strategy configs
│
├── tests/                     # Test suite (89% coverage)
├── artifacts/runs/            # Pipeline run outputs
├── datasets/                  # Input datasets
└── scripts/                   # Utility scripts
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ingest` | Load and normalize datasets into judging samples |
| `infer` | Run LLM inference to generate relevance judgements |
| `merge-infer-runs` | Combine multiple inference runs |
| `aggregate` | Apply ensemble voting (majority, average, random) |
| `evaluate` | Compute inter-rater agreement metrics |

### Example Workflow

```bash
# 1. Ingest a dataset
ingest --input-path datasets/llm_judge_challenge_experiment \
       --io-cfg llm_judge_challenge_to_db \
       --run-name my-ingest

# 2. Run inference with multiple models (using free models)
infer --model-cfg gemma-3n-e2b-it-free --provider openrouter \
      --prompt-template thomas-simple --io-cfg db_to_db --input my-ingest

infer --model-cfg gpt-oss-20b-free --provider openrouter \
      --prompt-template thomas-simple --io-cfg db_to_db --input my-ingest

infer --model-cfg mai-ds-r1-free --provider openrouter \
      --prompt-template thomas-simple --io-cfg db_to_db --input my-ingest

# 3. Aggregate predictions using majority vote
aggregate --aggregation-strategy majority_vote --io-cfg db_to_db \
          --input-run-names <run1> <run2> <run3>

# 4. Evaluate agreement with ground truth
evaluate --io-cfg db_aggregate_to_json --input-run-name <aggregate-run-name>
```

## Configuration

**Environment variables** (`.env`):
```
OPENROUTER_API_KEY=your-key-here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=llm_ensemble
POSTGRES_PASSWORD=llm_ensemble
POSTGRES_DB=llm_ensemble
```

**Model configs** (`configs/models/`): YAML files defining model parameters, pricing, and context windows.

**Prompt templates** (`configs/prompts/`): Customizable prompt templates for different judging strategies.

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- OpenRouter API key (or other LLM provider credentials)

## Testing

```bash
# Run all tests
make test

# Run with coverage
pytest --cov=llm_ensemble --cov-report=term-missing

# Run specific pipeline tests
pytest tests/ingest/
pytest tests/infer/
pytest tests/aggregate/
```

## Architecture

The project follows **Hexagonal Architecture** (Ports & Adapters):

- **Domain Layer**: Pure business logic (entities, voting strategies)
- **Application Layer**: Use cases and orchestration
- **Adapters**: CLI (driving), Database/API (driven)
- **Ports**: Interfaces between layers

This ensures clean separation of concerns, testability, and easy extension with new LLM providers or voting strategies.
