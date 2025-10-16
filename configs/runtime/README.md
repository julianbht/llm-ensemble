# Runtime Environment Configurations

Environment-specific settings loaded at CLI startup using a **two-layer approach** with python-dotenv:
1. **Root `.env`** (gitignored) — secrets and local overrides
2. **`configs/runtime/*.env`** (committed) — environment-specific non-secret defaults

This follows 12-factor app principles by separating environment-specific configuration from code while keeping secrets out of version control.

## How It Works

1. **CLI imports** trigger `load_runtime_config()` at module load
2. **Layer 1:** Load root `.env` file (if exists) for secrets and local overrides
3. **Layer 2:** Read `APP_ENV` environment variable (defaults to "dev")
4. **Layer 3:** Load `configs/runtime/{APP_ENV}.env` for environment-specific defaults
5. **Environment variables** set from files (only if not already set)
6. **Logger and other components** read these env vars

## Files

### Committed (configs/runtime/)
- **`dev.env`** — Development environment (human-readable logs, DEBUG level)
- **`prod.env`** — Production environment (JSON logs, WARNING level)
- **`ci.env`** — CI/CD environment (JSON logs, INFO level)

### Gitignored (project root)
- **`.env`** — Secrets (API keys, tokens) and local overrides (not in git)

## Precedence

Environment variables are set in this order (highest to lowest):
1. **Already-set shell environment variables** (manual overrides)
2. **Root `.env` file** (secrets and local overrides, gitignored)
3. **Runtime config file** (`configs/runtime/{APP_ENV}.env`, committed)
4. **Code defaults**

## Usage

```bash
# Development (default)
ingest --adapter llm-judge --data-dir ./data

# Production
APP_ENV=prod ingest --adapter llm-judge --data-dir ./data

# Override specific variable (shell env takes precedence)
APP_ENV=prod LOG_LEVEL=INFO ingest --adapter llm-judge --data-dir ./data

# Store secrets in root .env (gitignored)
echo "OPENROUTER_API_KEY=sk-..." >> .env
echo "HF_TOKEN=hf_..." >> .env
```

## Example Root .env File

```bash
# .env (gitignored - do NOT commit this file)
# Place in project root for secrets and local overrides

# API Keys (secrets)
OPENROUTER_API_KEY=sk-or-v1-...
OLLAMA_BASE_URL=http://localhost:11434
HF_TOKEN=hf_...

# Local overrides (optional)
# Uncomment to override environment-specific defaults
# LOG_LEVEL=DEBUG
# LOG_FORMAT=human
```

## Run Organization

Runs are organized by type for reproducibility:

- **Test runs:** `artifacts/runs/<cli_name>/test/<run_id>/`
  - Used for development, experimentation, quick tests
  - Not tracked in git, can be deleted freely

- **Official runs:** `artifacts/runs/<cli_name>/official/<run_id>/`
  - Used for thesis results, paper figures, benchmarks
  - Can be git-tracked for full reproducibility
  - Use `--official` flag in CLIs (e.g., `infer --official`)
  - Include `--notes` to document experiment purpose
