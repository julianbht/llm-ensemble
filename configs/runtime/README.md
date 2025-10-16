# Runtime Environment Configurations

Environment-specific settings loaded at CLI startup based on `APP_ENV` variable.

## How It Works

1. **CLI imports** trigger `load_runtime_config()` at module load
2. **Loader reads** `APP_ENV` environment variable (defaults to "dev")
3. **Corresponding `.env` file** loaded (e.g., `dev.env`, `prod.env`, `ci.env`)
4. **Environment variables** set from file (only if not already set)
5. **Logger and other components** read these env vars

## Files

- **`dev.env`** — Development environment (human-readable logs, DEBUG level)
- **`prod.env`** — Production environment (JSON logs, WARNING level)
- **`ci.env`** — CI/CD environment (JSON logs, INFO level)

## Precedence

Environment variables are set in this order (highest to lowest):
1. **Already-set env vars** (manual overrides, shell environment)
2. **Runtime config file** (`configs/runtime/{APP_ENV}.env`)
3. **Code defaults**

## Usage

```bash
# Development (default)
ingest --adapter llm-judge --data-dir ./data

# Production
APP_ENV=prod ingest --adapter llm-judge --data-dir ./data

# Override specific variable
APP_ENV=prod LOG_LEVEL=INFO ingest --adapter llm-judge --data-dir ./data
```
