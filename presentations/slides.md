---
title: LLM Ensemble Overview

# Presentation theme (overall look)
# Options: black, white, league, beige, sky, night, serif, simple, solarized, blood, moon
theme: white

# Code syntax highlighting theme
# Dark: monokai, atom-one-dark, vs2015, dracula, nord, zenburn
# Light: github, atom-one-light, vs, stackoverflow-light
highlightTheme: monokai

# Reveal.js configuration options
revealOptions:
  transition: 'none'      # none, fade, slide, convex, concave, zoom
  controls: true          # Show navigation controls
  progress: true          # Show progress bar
  slideNumber: 'false'    # Show slide numbers
  hash: true              # Update URL hash for each slide
---

# LLM Ensemble

A framework for creating relevance judgements for information retrieval datasets using an ensemble of LLMs

---

## What is LLM Ensemble?

- Framework for evaluating LLM performance on ranking tasks
- Supports single models or ensembles
- Built for the LLM4Eval challenge
- Hexagonal architecture (Ports & Adapters)

---

## Key Features

- **Multiple LLM Support**: OpenAI, OpenRouter, and more
- **Ensemble Voting**: Majority, average, random strategies
- **Evaluation Metrics**: Cohen's Kappa, Krippendorff's Alpha
- **PostgreSQL Storage**: Track all runs and results
- **Observability**: Grafana + Loki integration

---

## Architecture

```
┌─────────────────────────────────────┐
│     adapters/driving/               │
│     (CLI - Typer apps)              │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     application/                    │
│     (Use cases & orchestration)     │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     domain/                         │
│     (Business logic & entities)     │
└─────────────────────────────────────┘
```

---

## CLI Tools

Five main commands:

1. **ingest** - Load datasets
2. **infer** - Run LLM inference
3. **merge-infer-runs** - Combine multiple runs
4. **aggregate** - Apply ensemble voting
5. **evaluate** - Compute agreement metrics

---

## Quick Start

```bash
# Setup
make install-dev
make db && make db-init

# Run inference
infer --model gpt-4 \
      --prompt thomas-et-al-prompt \
      --io json \
      --input samples.json

# Aggregate results
aggregate --strategy majority \
          --runs run1,run2,run3

# Evaluate
evaluate --run-id <run-id>
```

---

## Example Workflow

```bash
# 1. Ingest dataset
ingest --dataset llm4eval

# 2. Run multiple models
infer --model gpt-4 --prompt default
infer --model claude-3 --prompt default
infer --model llama-3 --prompt default

# 3. Create ensemble
aggregate --strategy majority --runs run1,run2,run3

# 4. Evaluate agreement
evaluate --run-id ensemble-run-id
```

---

## Ensemble Strategies

- **Majority Voting**: Most common label wins
- **Average**: Mean of numeric scores
- **Random**: Random selection from disagreements

Results tracked with:
- Cohen's Kappa
- Krippendorff's Alpha

---

## Technology Stack

- **Python 3.11+** with Pydantic for validation
- **Typer** for CLI interfaces
- **SQLAlchemy 2.0** (async) for database
- **PostgreSQL** for persistence
- **Docker Compose** for infrastructure
- **Grafana + Loki** for observability

---

## Observability

Real-time monitoring with structured logs:

```bash
make observability
# Visit http://localhost:3000
```

- Track all CLI operations
- Monitor API calls and costs
- Debug failed runs
- Query logs with LogQL

---

## Project Structure

```
llm-ensemble/
├── src/llm_ensemble/
│   ├── ingest/          # Dataset loading
│   ├── infer/           # LLM inference
│   ├── aggregate/       # Ensemble voting
│   └── evaluate/        # Metrics computation
├── configs/models/      # Model configurations
├── artifacts/runs/      # Run outputs
└── datasets/            # LLM4Eval data
```

---

## Testing

```bash
make test              # All tests
pytest -m unit        # Fast unit tests
pytest -m integration # Integration tests
```

Test markers:
- `unit` - Fast, isolated
- `integration` - With I/O
- `slow` - API calls
- `requires_api` - Needs credentials

---

## Questions?

Documentation: [CLAUDE.md](../CLAUDE.md)

Repository: Your repo URL here

---

## Thank You!

🚀 Ready to evaluate LLMs at scale
