# Structured JSON Logging

## Overview

All LLM Ensemble CLIs now use **structlog** for structured JSON logging. Logs are machine-readable, queryable, and automatically enriched with reproducibility metadata.

## Architecture

**Dual output approach:**
- **`typer.echo()`** → User-facing CLI messages (stderr)
- **`structlog` logger** → Structured operational logs (stderr + file)

Both coexist: users see readable messages, while structured logs enable analysis.

## Log Format

Every log entry is a JSON object with:

```json
{
  "event": "inference_started",
  "cli": "infer",
  "run_id": "20251015_134402_phi3",
  "git_sha": "85bf941",
  "model": "phi3-mini",
  "num_samples": 100,
  "level": "info",
  "logger": "infer",
  "timestamp": "2025-10-15T17:44:02.637529Z"
}
```

### Standard Fields (all logs)

- `event`: Event name (e.g., `inference_started`, `parse_failed`)
- `cli`: CLI name (`ingest`, `infer`, `aggregate`, `evaluate`)
- `run_id`: Unique run identifier
- `git_sha`: Git commit SHA (reproducibility)
- `level`: Log level (`debug`, `info`, `warning`, `error`)
- `logger`: Logger name (typically same as `cli`)
- `timestamp`: ISO 8601 timestamp

### Event-Specific Fields

Each event adds relevant context:

**`inference_success`:**
```json
{
  "event": "inference_success",
  "query_id": "q1",
  "doc_id": "d123",
  "label": 2,
  "latency_ms": 234.5
}
```

**`inference_failed`:**
```json
{
  "event": "inference_failed",
  "query_id": "q2",
  "doc_id": "d456",
  "error": "Connection timeout",
  "error_type": "TimeoutError"
}
```

## Usage in CLIs

### Ingest CLI

**Events:**
- `ingest_started` — Dataset ingestion begins
- `ingest_completed` — Ingestion finished successfully
- `ingest_failed` — Exception during ingestion

**Example:**
```bash
ingest --adapter llm-judge --data-dir ./data --limit 100
```

**Output (stderr):**
```
Run ID: 20251015_134402_llm-judge
Output: artifacts/runs/ingest/20251015_134402_llm-judge/samples.ndjson
{"event": "ingest_started", "cli": "ingest", "run_id": "20251015_134402_llm-judge", ...}
Wrote 100 examples
{"event": "ingest_completed", "sample_count": 100, ...}
```

**Log file:** `artifacts/runs/ingest/20251015_134402_llm-judge/logs.jsonl`

### Infer CLI

**Events:**
- `inference_started` — Inference run begins
- `inference_success` — Individual judgement succeeded (debug level)
- `inference_failed` — Individual judgement failed (debug level)
- `inference_completed` — Inference run finished

**Example:**
```bash
infer --model phi3-mini --input artifacts/runs/ingest/xyz/samples.ndjson
```

**Log file:** `artifacts/runs/infer/<run_id>/logs.jsonl`

## Log Analysis

### Using Python

Load logs into pandas for analysis:

```python
import pandas as pd

# Load logs
df = pd.read_json("artifacts/runs/infer/xyz/logs.jsonl", lines=True)

# Filter to successes
successes = df[df['event'] == 'inference_success']

# Latency stats
print(successes['latency_ms'].describe())

# Find slow queries
slow = successes[successes['latency_ms'] > 500]
```

### Using the Analysis Script

```bash
python scripts/analyze_logs.py artifacts/runs/ingest/xyz/logs.jsonl
```

**Output:**
```
=== Ingest Run Analysis: 20251015_134402_llm-judge ===

Total log entries: 2
Events: ingest_started, ingest_completed

Dataset: llm-judge
Data dir: data
Limit: 5
Samples processed: 5
Git SHA: 85bf941
Started: 2025-10-15T17:44:02.637529Z
Completed: 2025-10-15T17:44:02.703081Z
```

### Using jq (if installed)

```bash
# Count errors
cat logs.jsonl | jq 'select(.level == "error")' | wc -l

# Average latency
cat logs.jsonl | jq 'select(.event == "inference_success") | .latency_ms' | \
  awk '{sum+=$1; n++} END {print sum/n}'

# Filter by run_id
cat artifacts/runs/*/*/logs.jsonl | jq 'select(.run_id == "xyz")'

# Pretty-print all logs
cat logs.jsonl | jq '.'
```

## Implementation

### Shared Library

`src/llm_ensemble/libs/logging/json_logger.py` provides `configure_logging()`:

```python
from llm_ensemble.libs.logging import configure_logging

logger = configure_logging(
    cli_name="infer",
    run_id=run_id,
    log_file=Path("artifacts/runs/infer/xyz/logs.jsonl"),
    git_sha="85bf941"
)

logger.info("inference_started", model="phi3", num_samples=100)
logger.debug("inference_success", query_id="q1", latency_ms=234.5)
logger.error("inference_failed", query_id="q2", error="timeout")
```

### Log Levels

- `DEBUG` — Per-sample events (inference_success, inference_failed)
- `INFO` — Run lifecycle events (started, completed)
- `WARNING` — Unusual but non-fatal conditions
- `ERROR` — Failures and exceptions

Default level: `INFO` (change via `log_level` parameter)

## Reproducibility

Every log record includes:
- `run_id` — Unique run identifier
- `git_sha` — Exact code version
- `cli` — Which CLI produced this log

Logs are saved to `artifacts/runs/<cli>/<run_id>/logs.jsonl` and referenced in `manifest.json`.

## Best Practices

**Do:**
- Log structured data: `logger.info("event_name", key=value)`
- Use meaningful event names: `inference_started`, `parse_failed`
- Include relevant context: query_id, doc_id, model, etc.
- Keep `typer.echo()` for user messages

**Don't:**
- Log strings: ~~`logger.info(f"Started with {model}")`~~
- Use print(): Use `typer.echo()` instead
- Log sensitive data (API keys, credentials)

## Future Enhancements

- Log level control via CLI flag (`--log-level debug`)
- Log aggregation across multiple runs
- Structured error tracking (warnings, retries)
- Cost tracking (tokens, API calls)
