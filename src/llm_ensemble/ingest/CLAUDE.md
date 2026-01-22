# Ingest Pipeline

Reads raw IR datasets and normalizes them into `JudgingSample` entities for downstream inference.

## Data Flow

```
Raw Dataset → ForInput (reader) → NormalizedDataset → ForOutput (writer) → Storage
```

## Key Entities

- **JudgingSample** - Core unit: query + document + gold relevance score
- **NormalizedDataset** - Collection of samples with dataset metadata
- **IngestRun** - Complete run record with config and results

## Ports

**Driven (dependencies):**
- `ForInput` - Read raw datasets, return `NormalizedDataset`
- `ForOutput` - Write `IngestRun` to storage (JSON/DB)

**Driving (entry points):**
- `ForRunningIngest` - Application interface called by CLI

## Directory Structure

```
ingest/
├── adapters/
│   ├── driving/ingest_cli.py      # CLI entry point
│   └── driven/io/                  # I/O adapters (JSON, DB)
├── application/
│   ├── ingest_application.py       # Main use case
│   └── ports/                      # Interface definitions
├── domain/entities/                # Domain models (Pydantic)
└── startup/dependency_configurator.py  # DI wiring
```

## CLI Usage

```bash
ingest --input-path ./data.json --io json --run-name my-run
```

## Adding New Dataset Formats

1. Create reader implementing `ForInput` in `adapters/driven/io/`
2. Register in `io_factory.py`
3. Add IO config option in `libs/cli/params/ingest.py`
