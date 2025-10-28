# Infer CLI

The **infer** CLI runs LLM inference on judging examples and outputs structured judgements.

## Architecture

Follows hexagonal architecture (ports & adapters) with strict separation of concerns:

### Layers

1. **CLI Layer** (`infer_cli.py`)
   - Typer entrypoint that parses arguments and delegates to orchestrator
   - No business logic - pure wiring

2. **Orchestrator Layer** (`orchestrator.py`)
   - Infrastructure coordination: run management, logging, manifests
   - Loads configurations, instantiates adapters via factories
   - Delegates business logic to domain service

3. **Domain Layer** (`domain/`)
   - `inference_service.py`: Pure business logic coordinating read → infer → write pipeline
   - No I/O dependencies - depends only on port abstractions
   - Fully testable without external dependencies

4. **Ports Layer** (`ports/`)
   - Abstract base classes (ABCs) defining infrastructure contracts
   - `LLMProvider`: LLM inference interface
   - `ExampleReader`: Reading JudgingExample records
   - `JudgementWriter`: Writing ModelJudgement records
   - `PromptBuilder`: Building prompts from templates
   - `ResponseParser`: Parsing LLM responses

5. **Adapters Layer** (`adapters/`)
   - Concrete implementations of ports
   - `io/`: File I/O adapters (NDJSON reader/writer)
   - `providers/`: LLM provider implementations (OpenRouter, Ollama, HuggingFace)
   - `prompts/`: Prompt builders (Jinja2)
   - `parsers/`: Response parsers (JSON)
   - Factories: `io_factory.py`, `provider_factory.py`, `prompt_builder_factory.py`, `response_parser_factory.py`

6. **Schemas** (`schemas/`)
   - Pydantic models for data validation
   - `model_judgement_schema.py`: ModelJudgement output
   - `model_config_schema.py`: Model configuration
   - `prompt_config_schema.py`: Prompt configuration
   - `io_config_schema.py`: I/O format configuration

7. **Config Loaders** (`config_loaders/`)
   - YAML configuration loading and validation
   - `model_config_loader.py`, `prompt_config_loader.py`, `io_config_loader.py`

## Usage

### Basic Examples

```bash
# Basic usage (uses default prompt and I/O format)
infer --model gpt-oss-20b --input artifacts/runs/ingest/<run_id>/samples.ndjson

# Limit processing to first 10 examples
infer --model gpt-oss-20b --input samples.ndjson --limit 10

# Explicit prompt and I/O format
infer --model phi3-mini --input samples.ndjson --prompt thomas-et-al-prompt --io ndjson

# Override model parameters
infer --model gpt-oss-20b --input samples.ndjson \
      --override default_params.temperature=0.7 \
      --override default_params.max_tokens=512

# Override prompt variables
infer --model phi3-mini --input samples.ndjson \
      --override variables.role=false \
      --override variables.aspects=true

# Official run with logs and notes
infer --model gpt-oss-20b --input samples.ndjson --save-logs --official \
      --notes "Baseline experiment with temperature=0"
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

All behavior is configured via YAML files in `configs/`. CLI flags reference config names, not paths.

#### Model Configs (`configs/models/*.yaml`)

Define provider, model name, context window, and default parameters.

- **OpenRouter:** `provider: openrouter` + `openrouter_model_id`
- **Ollama:** `provider: ollama` + local model name
- **HuggingFace:** `provider: huggingface` + `hf_endpoint_url` or `hf_model_name`

Example: `--model gpt-oss-20b` loads `configs/models/gpt-oss-20b.yaml`

#### Prompt Configs (`configs/prompts/*.yaml`)

Define Jinja2 template file, variables, and bundled response parser.

- `template_file`: Jinja2 template path (relative to `configs/prompts/`)
- `variables`: Template variables (can be overridden via `--override`)
- `prompt_builder`: Adapter name (e.g., `jinja_prompt_builder`)
- `response_parser`: Adapter name (e.g., `json_response_parser`)

Example: `--prompt thomas-et-al-prompt` loads `configs/prompts/thomas-et-al-prompt.yaml`

#### I/O Configs (`configs/io/*.yaml`)

Define bundled reader and writer adapters for specific formats.

- `reader`: Adapter name (e.g., `ndjson_example_reader`)
- `writer`: Adapter name (e.g., `ndjson_judgement_writer`)

Example: `--io ndjson` loads `configs/io/ndjson.yaml`

**Note:** All configs support runtime overrides via `--override key=value`

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

## Extending the Infer CLI

### Adding a New Provider

To add support for a new LLM provider:

1. **Create adapter** in `adapters/providers/new_provider_adapter.py`
   - Implement `LLMProvider` port (ABC) from `ports/llm_provider.py`
   - Implement `infer()` method that yields `ModelJudgement` objects

2. **Update factory** in `adapters/provider_factory.py`
   - Add case for your provider in `get_provider()` function
   - Instantiate your adapter when `model_config.provider == "new_provider"`

3. **Add model configs** to `configs/models/` with `provider: new_provider`

All other components (prompts, parsers, I/O, domain service) are provider-agnostic and reusable.

### Adding a New I/O Format

To add support for a new file format:

1. **Create adapters** in `adapters/io/`
   - Reader implementing `ExampleReader` port
   - Writer implementing `JudgementWriter` port

2. **Update factory** in `adapters/io_factory.py`
   - Add cases in `get_example_reader()` and `get_judgement_writer()`

3. **Add I/O config** to `configs/io/new_format.yaml`
   - Specify reader and writer adapter names

### Adding a New Prompt Format

To add support for a new prompt format:

1. **Create builder** in `adapters/prompts/` implementing `PromptBuilder` port
2. **Create parser** in `adapters/parsers/` implementing `ResponseParser` port
3. **Update factories** in `prompt_builder_factory.py` and `response_parser_factory.py`
4. **Add prompt config** to `configs/prompts/` bundling builder + parser

The hexagonal architecture ensures all extensions are isolated and testable.

## Data Flow

```
CLI (infer_cli.py)
  ↓ parse args
Orchestrator (orchestrator.py)
  ↓ load configs, create run dir, instantiate adapters via factories
Domain Service (InferenceService)
  ↓ coordinate pipeline
  ├─→ ExampleReader.read(input_path) → [JudgingExample, ...]
  ├─→ LLMProvider.infer(examples) → [ModelJudgement, ...]
  │     ├─→ PromptBuilder.build() → prompt text
  │     ├─→ HTTP call to provider API
  │     └─→ ResponseParser.parse() → ModelJudgement
  └─→ JudgementWriter.write(judgement)
  ↓
Orchestrator writes manifest
  ↓
artifacts/runs/infer/<run_id>/
  ├── judgements.ndjson
  ├── manifest.json
  └── run.log (if --save-logs)
```

**Key principles:**
- **Domain service** coordinates workflow using only port abstractions
- **Ports** define contracts, adapters provide implementations
- **Factories** instantiate adapters based on configuration
- **Orchestrator** handles infrastructure (logging, manifests, run management)
