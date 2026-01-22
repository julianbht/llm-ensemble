# Evaluate Pipeline

Computes inter-rater agreement metrics between LLM predictions and ground truth labels.

## Data Flow

```
InferRun or AggregateRun → Extract (ground_truth, predictions) → Metrics → EvaluateRun
```

Reads predictions from an infer or aggregate run, compares against gold labels, and outputs metric results.

## Key Entities

- **MetricResult** - Standardized metric output: name, value, sample_size, interpretation
- **EvaluationData** - Extracted ground truth and predictions for metric computation
- **EvaluateRun** - Run record with config and metric results

## Available Metrics

- `cohens_kappa` - Inter-rater agreement accounting for chance (scikit-learn)
  - Range: -1 to 1 (1 = perfect, 0 = chance, <0 = worse than chance)
  - Interpretation: slight, fair, moderate, substantial, almost perfect
- `krippendorffs_alpha` - Inter-rater reliability for incomplete data
  - Handles missing predictions (failed parses)

## Ports

**Driven (dependencies):**
- `ForInput` - Read predictions from infer or aggregate runs
- `ForOutput` - Write `EvaluateRun` results
- `ForComputingMetrics` - Compute individual metrics (multiple adapters)

**Driving (entry points):**
- `ForRunningEvaluation` - Application interface called by CLI

## Directory Structure

```
evaluate/
├── adapters/
│   ├── driving/evaluate_cli.py     # CLI entry point
│   └── driven/
│       ├── metrics/                # Metric implementations
│       │   ├── cohens_kappa.py
│       │   └── krippendorffs_alpha.py
│       └── io/                     # I/O adapters (JSON, DB readers)
├── application/
│   ├── evaluation_application.py   # Main use case
│   └── ports/                      # Interface definitions
├── domain/
│   ├── entities/                   # Domain models (Pydantic)
│   └── evaluation_data_builder.py  # Extract data for metrics
└── startup/dependency_configurator.py
```

## CLI Usage

```bash
# Evaluate an inference run
evaluate --io json --input-run-name my-infer-run

# Evaluate an aggregate run
evaluate --io json --input-run-name my-aggregate-run
```

## Output

The evaluate run produces a summary with all computed metrics:
```json
{
  "metric_results": [
    {
      "name": "cohens_kappa",
      "value": 0.72,
      "sample_size": 500,
      "interpretation": "substantial"
    }
  ]
}
```

## Adding New Metrics

1. Create adapter implementing `ForComputingMetrics` in `adapters/driven/metrics/`
2. Return `MetricResult` with standardized fields (name, value, interpretation, etc.)
3. Register in `metric_factory.py`
