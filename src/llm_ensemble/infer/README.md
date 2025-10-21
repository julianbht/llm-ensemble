# Infer CLI

The **infer** CLI runs LLM inference on judging examples and outputs structured judgements.

## Architecture

Follows **clean architecture / ports & adapters** pattern (Alistair Cockburn's Hexagonal Architecture):

### Core Layers

- **Domain** (`domain/`) — Pure business logic with no I/O dependencies
  - `inference_service.py`: Orchestrates inference pipeline using port abstractions

- **Ports** (`ports/`) — Abstract interfaces (ABCs) defining contracts
  - `llm_provider.py`: LLM inference abstraction
  - `example_reader.py`: Input reading abstraction
  - `judgement_writer.py`: Output writing abstraction
  - `prompt_builder.py`: Prompt construction abstraction
  - `response_parser.py`: Response parsing abstraction

- **Adapters** (`adapters/`) — Concrete implementations of ports
  - **Providers** (`adapters/providers/`)
    - `openrouter_adapter.py`: OpenRouter API implementation
    - `ollama_adapter.py`: Ollama local inference implementation
    - `huggingface_adapter.py`: HuggingFace Inference Endpoints implementation
  - **I/O** (`adapters/io/`)
    - `ndjson_example_reader.py`: NDJSON input reader
    - `ndjson_judgement_writer.py`: NDJSON output writer

- **Prompt Builders** (`prompt_builders/`) — Prompt construction adapters
  - `thomas.py`: Thomas et al. prompt format implementation
  - Each builder implements `PromptBuilder` port

- **Response Parsers** (`response_parsers/`) — Response parsing adapters
  - `thomas.py`: Thomas et al. JSON response parser
  - Each parser implements `ResponseParser` port

- **Schemas** (`schemas/`) — Pydantic models for data contracts
  - `model_judgement_schema.py`: Output judgement records
  - `model_config_schema.py`: Model configuration
  - `prompt_config_schema.py`: Prompt configuration
  - `io_config_schema.py`: I/O format configuration

- **Config Loaders** (`config_loaders/`) — YAML configuration loading
  - Model, prompt, and I/O config readers

- **CLI layer** (`infer_cli.py`) — Typer entrypoint that wires adapters to domain service

### Dependency Flow

```
CLI → Domain Service → Ports (ABC) ← Adapters (Concrete)
```

The domain layer depends only on port abstractions, never on concrete adapters. This enables:
- **Testing**: Mock implementations for isolated unit tests
- **Substitutability**: Swap providers, parsers, I/O formats without changing domain logic
- **Clarity**: Explicit boundaries between business logic and infrastructure

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

## Extending the System

### Adding a New LLM Provider

1. **Create adapter** implementing `LLMProvider` port:
   ```python
   # adapters/providers/my_provider_adapter.py
   from llm_ensemble.infer.ports import LLMProvider
   
   class MyProviderAdapter(LLMProvider):
       def infer(self, examples, model_config, prompt_template_name, prompts_dir):
           # Implementation
           yield ModelJudgement(...)
   ```

2. **Update provider factory** (`adapters/provider_factory.py`) to instantiate your adapter

3. **Add model configs** to `configs/models/` with `provider: my_provider`

### Adding a New Prompt Format

1. **Create builder adapter** implementing `PromptBuilder` port:
   ```python
   # prompt_builders/my_format.py
   from llm_ensemble.infer.ports import PromptBuilder
   
   class MyFormatPromptBuilder(PromptBuilder):
       def build(self, template, example):
           return template.render(...)
   ```

2. **Create parser adapter** implementing `ResponseParser` port:
   ```python
   # response_parsers/my_format.py
   from llm_ensemble.infer.ports import ResponseParser
   
   class MyFormatResponseParser(ResponseParser):
       def parse(self, raw_text):
           # Extract label and warnings
           return (label, warnings)
   ```

3. **Add prompt config** to `configs/prompts/my-format-prompt.yaml`:
   ```yaml
   name: my-format-prompt
   prompt_template: my-format-prompt  # References my-format-prompt.jinja
   prompt_builder: my_format          # References prompt_builders/my_format.py
   response_parser: my_format         # References response_parsers/my_format.py
   ```

4. **Create Jinja2 template** at `configs/prompts/my-format-prompt.jinja`

### Adding a New I/O Format

1. **Create reader adapter** implementing `ExampleReader` port:
   ```python
   # adapters/io/my_format_example_reader.py
   from llm_ensemble.infer.ports import ExampleReader
   
   class MyFormatExampleReader(ExampleReader):
       def read(self, input_path, limit=None):
           # Return list of JudgingExample
           return examples
   ```

2. **Create writer adapter** implementing `JudgementWriter` port:
   ```python
   # adapters/io/my_format_judgement_writer.py
   from llm_ensemble.infer.ports import JudgementWriter
   
   class MyFormatJudgementWriter(JudgementWriter):
       def write(self, judgement):
           # Write judgement to file
           pass
   ```

3. **Update I/O factory** (`adapters/io_factory.py`) to instantiate your adapters

4. **Add I/O config** to `configs/io/my-format.yaml`:
   ```yaml
   io_format: my_format
   reader: my_format_example_reader
   writer: my_format_judgement_writer
   ```

All extensions follow the ports & adapters pattern: implement the port interface, register in the factory, add configuration. The domain service remains unchanged.
