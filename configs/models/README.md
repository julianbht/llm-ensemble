# Model Configurations

Model configs for LLM judge inference. One YAML file per model.

## How It Works

1. **Infer CLI** receives `--model <model_id>` flag
2. **Config loader** reads `<model_id>.yaml` from this directory
3. **Inference router** uses config to call the appropriate provider (OpenRouter, HuggingFace, Ollama)
4. **Model parameters** (temperature, max_tokens) applied during inference

## Config Structure

```yaml
model_id: gpt-oss-20b
provider: openrouter              # openrouter | hf | ollama
context_window: 8192

# Core inference parameters (explicit for discoverability)
temperature: 0.0                  # [0.0-2.0] Sampling temperature
max_tokens: 256                   # Maximum tokens to generate
seed: 42                          # For reproducibility (optional)
top_p: 0.95                       # Nucleus sampling (optional)
frequency_penalty: 0.0            # Reduce repetition (optional)
presence_penalty: 0.0             # Encourage diversity (optional)
stop: ["END"]                     # Stop sequences (optional)

# Output control
response_format:                  # Force structured output (optional)
  type: json_object

# Advanced/provider-specific parameters
additional_params:
  top_k: 50                       # Provider-specific advanced params

# Provider-specific fields
openrouter_model_id: openai/gpt-oss-20b:free   # For OpenRouter
# hf_endpoint_url: https://...                  # For HuggingFace
```

## Supported Providers

- **openrouter** — Requires `OPENROUTER_API_KEY` env var and `openrouter_model_id`
- **hf** — Requires `HF_TOKEN` and `hf_endpoint_url` or `hf_model_name`
- **ollama** — Requires local Ollama server (set `OLLAMA_BASE_URL`)

## Usage

```bash
infer --model gpt-oss-20b --input samples.ndjson
```
