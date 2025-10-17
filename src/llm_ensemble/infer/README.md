# Infer CLI

The **infer** CLI runs LLM inference on judging examples and outputs structured judgements.

## Architecture

Follows clean architecture / ports & adapters pattern with orchestrator coordination:

- **Orchestrator** (`orchestrator.py`) — Top-level coordination logic that wires components together

- **Schemas** (`schemas/`) — Pydantic models for data structures
  - `model_judgement.py`: Output judgement records
  - `model_config.py`: Model configuration schema

- **Prompts** (`prompts/`) — Prompt construction and template rendering
  - Template loading and variable interpolation

- **Parsers** (`parsers/`) — Response parsing logic
  - Extract structured data from LLM outputs

- **Providers** (`providers/`) — LLM provider adapters (I/O boundary)
  - `openrouter.py`: OpenRouter API adapter
  - `ollama.py`: Ollama local inference adapter
  - `huggingface.py`: HuggingFace Inference Endpoints adapter
  - `provider_router.py`: Routes to appropriate provider based on config

- **Config Loaders** (`config_loaders/`) — YAML configuration loading
  - Model and prompt config readers

- **CLI layer** (`infer_cli.py`) — Typer entrypoint that delegates to orchestrator

## Usage

### Basic Example

```bash
# Run inference with a model on samples
infer --model gpt-oss-20b --input artifacts/runs/ingest/<run_id>/samples.ndjson --limit 10

# Use a custom prompt template
infer --model phi3-mini --input samples.ndjson --prompt custom-prompt --limit 10
```

### Required Environment Variables

Depending on the provider, set the appropriate API credentials:

**OpenRouter models:**
```bash
export OPENROUTER_API_KEY="your_openrouter_api_key"
```

**HuggingFace models:**
```bash
export HF_TOKEN="your_huggingface_token"
```

**Ollama models:**
No credentials required (uses local Ollama instance)

### Configuration

#### Model Configs

Model configs are read from `configs/models/*.yaml`. The `model_id` field matches the config filename (e.g., `gpt-oss-20b.yaml` → `--model gpt-oss-20b`).

**OpenRouter example:**
```yaml
model_id: gpt-oss-20b
provider: openrouter
model_name: meta-llama/llama-3.1-405b-instruct
context_window: 8192
default_params:
  temperature: 0.0
  max_tokens: 512
```

**Ollama example:**
```yaml
model_id: phi3-mini
provider: ollama
model_name: phi3:mini
context_window: 4096
default_params:
  temperature: 0.0
  max_tokens: 256
```

**HuggingFace example:**
```yaml
model_id: tinyllama
provider: huggingface
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
context_window: 2048
default_params:
  temperature: 0.0
  max_tokens: 256
```

#### Prompt Configs

Prompt configs are read from `configs/prompts/*.yaml` and reference Jinja2 templates.

```yaml
name: thomas-et-al-prompt
template_file: thomas-et-al-prompt.jinja
description: |
  Thomas et al. search quality rater prompt with structured JSON output.
variables:
  role: true
  aspects: false
expected_output_format: json
response_parser: parse_thomas_response
```

## Input Format

Expects NDJSON with `JudgingExample` records from the ingest CLI:

```json
{"dataset":"llm-judge-2024","query_id":"q1","query_text":"...","docid":"d1","doc":"...","gold_relevance":1}
```

## Output Format

Produces NDJSON with `ModelJudgement` records written to `artifacts/runs/infer/<run_id>/judgements.ndjson`:

```json
{
  "model_id": "gpt-oss-20b",
  "provider": "openrouter",
  "query_id": "q1",
  "docid": "d1",
  "label": 2,
  "score": 2.0,
  "confidence": 0.95,
  "rationale": "The document directly addresses the query...",
  "raw_text": "{\"relevance\": 2, \"confidence\": 0.95, ...}",
  "latency_ms": 1234.5,
  "cost_estimate": 0.002,
  "retries": 0,
  "warnings": []
}
```

**Label values:**
- `0` = Not relevant
- `1` = Partially relevant
- `2` = Highly relevant

## Run Artifacts

All runs create organized output under `artifacts/runs/infer/<run_id>/`:

```
artifacts/runs/infer/<run_id>/
├── judgements.ndjson    # Output judgements
├── manifest.json        # Reproducibility metadata (git SHA, timestamps, etc.)
└── run.log             # Optional logs (if --save-logs used)
```

Use `--official` flag to save to `artifacts/runs/infer/official/<run_id>/` for git-tracked official runs.

## Extending with New Providers

To add support for a new provider:

1. Create `providers/new_provider.py` with `infer()` function that takes model config and judging example
2. Update `providers/provider_router.py` to route to your provider based on `provider` field
3. Add model configs to `configs/models/` with `provider: new_provider`

The prompts, parsers, and schemas are provider-agnostic and can be reused.
