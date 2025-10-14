# Infer CLI

The **infer** CLI runs LLM inference on judging examples and outputs structured judgements.

## Architecture

Follows clean architecture / ports & adapters pattern:

- **Domain layer** (`domain/`) — Pure Python logic with no I/O
  - `models.py`: Pydantic models for judgements and configs
  - `prompts.py`: Prompt construction logic
  - `parser.py`: Output parsing logic

- **Adapters layer** (`adapters/`) — I/O, APIs, file formats
  - `huggingface.py`: HuggingFace Inference Endpoints adapter
  - `config_loader.py`: YAML config file loader

- **CLI layer** (`infer_cli.py`) — Typer entrypoint that wires everything together

## Usage

### Basic Example

```bash
# Run inference with phi3-mini on samples
infer --model phi3-mini --input samples.ndjson --out judgements.ndjson --limit 10
```

### Required Environment Variables

For HuggingFace models:
```bash
export HF_TOKEN="your_huggingface_token"
```

### Configuration

Model configs are read from `configs/models/*.yaml`. Example:

```yaml
model_id: phi3-mini
provider: hf
context_window: 4096
default_params:
  temperature: 0.0
  max_tokens: 256
hf_model_name: microsoft/Phi-3-mini-4k-instruct
```

### Environment Variable Overrides (12-factor)

Override endpoint URLs via environment variables:

```bash
export HF_ENDPOINT_PHI3_MINI_URL="https://your-endpoint.aws.endpoints.huggingface.cloud"
```

Pattern: `HF_ENDPOINT_<MODEL_ID>_URL` (model ID uppercase with underscores)

## Input Format

Expects NDJSON with `JudgingExample` records from the ingest CLI:

```json
{"dataset":"llm-judge-2024","query_id":"q1","query_text":"...","docid":"d1","doc":"...","gold_relevance":1}
```

## Output Format

Produces NDJSON with `ModelJudgement` records:

```json
{
  "model_id": "phi3-mini",
  "provider": "hf",
  "query_id": "q1",
  "docid": "d1",
  "label": "relevant",
  "score": 0.9,
  "confidence": 0.9,
  "rationale": "...",
  "raw_text": "...",
  "latency_ms": 1234.5,
  "retries": 0,
  "warnings": []
}
```

## Extending with New Providers

To add support for a new provider (e.g., Ollama):

1. Create `adapters/ollama.py` with `iter_judgements()` function
2. Update `infer_cli.py` to route based on `provider` field in config
3. Add model configs to `configs/models/` with `provider: ollama`

The domain layer (prompts, parsing, models) is provider-agnostic.
