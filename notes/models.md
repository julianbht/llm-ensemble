How configs/models/ is Intended to Work in the Infer CLI

  The Design Pattern

  1. Model Config Files (configs/models/*.yaml):
    - Each file defines a model's configuration
    - Key fields:
        - model_id: Local identifier (e.g., "phi3-mini", "qwen-0.5b")
      - provider: Which service provides the model ("hf", "ollama", "openrouter")
      - default_params: Default inference parameters (temperature, max_tokens)
      - Provider-specific fields (e.g., hf_model_name, openrouter_model_id)
  2. CLI Workflow (from infer_cli.py:32-126):
  User runs: infer --model phi3-mini --input samples.ndjson
                         ↓
  CLI calls: load_model_config("phi3-mini", config_dir)
                         ↓
  Looks for: configs/models/phi3-mini.yaml
                         ↓
  Returns: ModelConfig object with all settings
                         ↓
  CLI routes to appropriate adapter based on provider field
  3. Missing Components (referenced but not implemented yet):
    - infer/adapters/config_loader.py - Loads YAML configs into ModelConfig objects
    - infer/adapters/huggingface.py - HuggingFace inference adapter (like our OpenRouter adapter)

  Current State

  What exists:
  - ✅ Model config files (phi3-mini.yaml, qwen-0.5b.yaml)
  - ✅ ModelConfig domain model (in infer/domain/models.py)
  - ✅ OpenRouter adapter (we just built it!)
  - ✅ Infer CLI skeleton

  What's missing:
  - ❌ config_loader.py - to load YAML → ModelConfig
  - ❌ huggingface.py adapter
  - ❌ Routing logic to pick the right adapter based on provider

  How It Should Work (Example)

  # User runs
  infer --model qwen-0.5b --input samples.ndjson

  # CLI loads configs/models/qwen-0.5b.yaml
  # Sees provider: openrouter
  # Calls openrouter.send_inference_request() with:
  #   - model_id from openrouter_model_id field
  #   - temperature/max_tokens from default_params

  The config system provides separation of concerns:
  - Configs define what models to use and their settings
  - Adapters define how to call each provider's API
  - CLI orchestrates everything together