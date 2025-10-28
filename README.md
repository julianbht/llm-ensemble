# LLM Ensemble

CLI-first research system for evaluating LLM-as-judge ensembles on information retrieval tasks.

## Four Core CLIs

1. **ingest** — Normalize raw IR datasets into `JudgingExample` records (NDJSON)
2. **infer** — Run multiple LLM judges over samples, writing per-model judgements
3. **aggregate** — Combine judgements using ensemble strategies (e.g., weighted majority vote)
4. **evaluate** — Compute metrics and generate HTML reports

All artifacts are written to `artifacts/runs/<cli_name>/<run_id>/` with manifests tracking git SHA, timestamps, and full reproducibility metadata.

## Quick Start

```bash
# Setup
python -m venv .venv
source .venv/bin/activate
make install-dev

# Run tests
make test

# Run individual CLIs
ingest --adapter llm-judge --data-dir ./data --limit 100
infer --model gpt-oss-20b --input artifacts/runs/ingest/<run_id>/samples.ndjson
```

## Development

```bash
make test              # Run all tests
make test-ingest       # Run ingest tests only
make test-infer        # Run infer tests only
make clean             # Remove test artifacts and cache
```

See [CLAUDE.md](./CLAUDE.md) for detailed architecture and development guidelines.
