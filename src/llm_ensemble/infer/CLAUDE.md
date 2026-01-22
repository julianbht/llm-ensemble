# Infer Pipeline

Runs LLM inference on `JudgingSample` entities to produce relevance judgements.

## Data Flow

```
JudgingSample → Prompt Builder → LLM Provider → Response Parser → LLMJudgement
```

Each sample is processed as: build prompt → call LLM → parse response → persist.

## Key Entities

- **LLMJudgement** - Complete judgement: sample + prompt + response + metrics + parsed score
- **LLMScore** - Parsed output: label, confidence, rationale
- **LLMInvocationMetrics** - Latency, cost, token counts, retry count
- **InferRun** - Run record with config and output

## Ports

**Driven (dependencies):**
- `ForInput` - Read `NormalizedDataset` from previous ingest run
- `ForOutput` - Write `LLMJudgement` results (streaming)
- `ForBuildingPrompts` - Render prompts from templates
- `ForInvokingLLM` - Call LLM APIs (OpenRouter, Ollama)
- `ForParsingResponses` - Extract scores from LLM responses

**Driving (entry points):**
- `ForRunningInference` - Application interface called by CLI

## Directory Structure

```
infer/
├── adapters/
│   ├── driving/
│   │   ├── infer_cli.py            # Main inference CLI
│   │   └── merge_infer_runs_cli.py # Merge multiple runs
│   └── driven/
│       ├── providers/              # LLM adapters (OpenRouter, Ollama)
│       ├── prompts/                # Prompt builder implementations
│       ├── parsers/                # Response parser implementations
│       └── io/                     # I/O adapters (JSON, DB)
├── application/
│   ├── inference_application.py    # Main use case
│   └── ports/                      # Interface definitions
├── domain/
│   ├── entities/                   # Domain models (Pydantic)
│   └── metrics.py                  # Agreement, cost calculations
└── startup/
    ├── dependency_configurator.py  # DI wiring
    └── config_loader.py            # Load YAML configs
```

## CLI Usage

```bash
# Basic inference
infer --model-cfg claude-sonnet-4 \
      --provider openrouter \
      --prompt-template thomas-simple \
      --io json \
      --input-run-name my-ingest-run

# With slicing (process samples 0-99)
infer ... --start-idx 0 --end-idx 100
```

## Configuration

- Model configs: `configs/models/*.yaml`
- Prompt templates: `configs/prompts/*.yaml`
- Retry configs: `configs/retries/*.yaml`

## Adding New Components

**New LLM Provider:**
1. Create adapter implementing `ForInvokingLLM` in `adapters/driven/providers/`
2. Register in `provider_factory.py`

**New Prompt Template:**
1. Create builder implementing `ForBuildingPrompts` in `adapters/driven/prompts/`
2. Add template YAML in `configs/prompts/`
3. Register in `prompt_factory.py`

**New Response Parser:**
1. Create parser implementing `ForParsingResponses` in `adapters/driven/parsers/`
2. Register in `parser_factory.py`
