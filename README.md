# LLM Ensemble

A pipeline for creating relevance judgements for information retrieval datasets using an ensemble of LLMs or a single model.

## Installation

```bash
# Clone and install
git clone https://github.com/julianbht/llm-ensemble.git
cd llm-ensemble
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set up environment
cp .env.example .env
# Edit .env with your API keys

# Start database (starts postgres in docker container)
make db
make db-init
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
│   ├── models/                # LLM model configurations
│   ├── prompts/               # Prompt templates
│   └── retries/               # Retry strategy configs
│
├── tests/                     # Test suite
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
ingest --input datasets/llm_judge_challenge_experiment \
       --io-cfg llm_judge_challenge_to_db \
       --run-name my-ingest

# 2. Run inference with multiple models (using free models)
infer --model-cfg gemma-3n-e2b-it-free --provider openrouter \
      --prompt-template thomas-advanced-trec --io-cfg db_to_db --input my-ingest

infer --model-cfg gpt-oss-20b-free --provider openrouter \
      --prompt-template thomas-advanced-trec --io-cfg db_to_db --input my-ingest

infer --model-cfg mai-ds-r1-free --provider openrouter \
      --prompt-template thomas-advanced-trec --io-cfg db_to_db --input my-ingest

# 3. Aggregate predictions using majority vote
aggregate --aggregation-strategy majority_vote_average --io-cfg db_to_db \
          --input <run1> --input <run2> --input <run3>

# 4. Evaluate agreement with ground truth
evaluate --io-cfg db_aggregate_to_json --input <aggregate-run-name>
```

## Configuration

**Environment variables** (`.env`):
```
OPENROUTER_API_KEY=your-key-here
DATABASE_URL=postgresql://llm_ensemble:llm_ensemble@localhost:5432/llm_ensemble
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
