# Aggregate Pipeline

Combines multiple inference runs into consensus relevance judgements using ensemble voting strategies.

## Data Flow

```
InferRun₁ ─┐
InferRun₂ ─┼─→ Group by sample → Aggregation Strategy → AggregatedVote
InferRun₃ ─┘
```

Judgements from multiple models are grouped by sample ID, then a voting strategy produces a final label.

## Key Entities

- **AggregatedVote** - Consensus result: final label, confidence, reasoning, source judgements
- **AggregationStrategy** - Strategy metadata (name, id)
- **AggregateRun** - Run record with config and aggregated dataset

## Voting Strategies

- `majority_vote` - Most common label wins, ties broken by lowest label
- `average_vote` - Mean of labels, rounded to nearest integer
- `random_vote` - Random selection from valid labels

## Ports

**Driven (dependencies):**
- `ForInput` - Read `InferRunOutput` from multiple inference runs
- `ForOutput` - Write `AggregateRun` results
- `ForAggregating` - Apply voting strategy to judgement groups

**Driving (entry points):**
- `ForRunningAggregation` - Application interface called by CLI

## Directory Structure

```
aggregate/
├── adapters/
│   ├── driving/aggregate_cli.py    # CLI entry point
│   └── driven/
│       ├── strategies/             # Voting strategy implementations
│       │   ├── majority_vote_adapter.py
│       │   ├── average_vote_adapter.py
│       │   └── random_vote_adapter.py
│       └── io/                     # I/O adapters (JSON, DB)
├── application/
│   ├── aggregation_application.py  # Main use case
│   └── ports/                      # Interface definitions
├── domain/
│   ├── entities/                   # Domain models (Pydantic)
│   ├── validation.py               # Input validation (fingerprint matching)
│   └── aggregate_statistics.py     # Tie/failure counting
└── startup/dependency_configurator.py
```

## CLI Usage

```bash
# Aggregate 3 inference runs with majority vote
aggregate --aggregation-strategy majority_vote \
          --io json \
          --input-run-names run1 run2 run3

# Using average voting
aggregate --aggregation-strategy average_vote \
          --io json \
          --input-run-names run1 run2 run3
```

## Validation

Before aggregating, the pipeline validates:
- All input runs have matching `sample_fingerprint` (same samples in same order)
- All runs are complete (no missing judgements)

## Adding New Strategies

1. Create adapter implementing `ForAggregating` in `adapters/driven/strategies/`
2. Register in `aggregation_strategy_factory.py`
3. Add CLI option in `libs/cli/params/aggregate.py`
