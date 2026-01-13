#!/usr/bin/env python3
"""
Extract cost and latency data from inference runs for visualization.

Queries the database to aggregate cost and latency metrics by model/ensemble.
Outputs structured JSON for downstream plotting.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from sqlalchemy import func
from sqlalchemy.orm import Session

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    InferRunConfigORM,
    InferRunOutputORM,
    LLMJudgementORM,
    ModelConfigORM,
)

# Configuration
OUTPUT_FILE = Path("artifacts/other/cost_latency_comparison.json")

# Specify which inference runs to include
# Format: List of run_name patterns or full run names
INCLUDE_RUNS = [
    # Ensemble runs:
    "ensemble-1-google-gemma-3-4b-it",
    "ensemble-2-ministral-3b-2515",
    "ensemble-3-phi-4-multimodal-instruct",
    "ensemble-4-meta-llama-3.2-3b-instruct-start",
    "ensemble-5-ui-tars-1.5-7b-start",
    # Single model reference runs:
    "reference-ensemble-gpt-5-1-all-samples-start",
]

# If empty, include all runs
INCLUDE_ALL_RUNS = False


def extract_run_metrics(session: Session) -> List[Dict[str, Any]]:
    """Extract cost and latency metrics from inference runs.

    Args:
        session: Database session

    Returns:
        List of run metrics with aggregated cost and latency
    """
    query = (
        session.query(
            InferRunORM.run_name,
            ModelConfigORM.name.label("model_name"),
            ModelConfigORM.name_hint.label("model_hint"),
            func.sum(LLMJudgementORM.latency_ms).label("total_latency_ms"),
            func.avg(LLMJudgementORM.latency_ms).label("avg_latency_ms"),
            func.sum(
                func.coalesce(
                    LLMJudgementORM.actual_cost_usd, LLMJudgementORM.cost_estimate_usd
                )
            ).label("total_cost_usd"),
            func.avg(
                func.coalesce(
                    LLMJudgementORM.actual_cost_usd, LLMJudgementORM.cost_estimate_usd
                )
            ).label("avg_cost_usd"),
            func.count(LLMJudgementORM.id).label("num_judgements"),
        )
        .join(InferRunORM.infer_run_output)
        .join(InferRunOutputORM.llm_judgements)
        .join(InferRunORM.infer_run_config)
        .join(InferRunConfigORM.model_config)
        .filter(InferRunOutputORM.finished == True)
    )

    # Apply run name filters if specified
    if not INCLUDE_ALL_RUNS and INCLUDE_RUNS:
        query = query.filter(InferRunORM.run_name.in_(INCLUDE_RUNS))

    query = query.group_by(
        InferRunORM.run_name,
        ModelConfigORM.name,
        ModelConfigORM.name_hint,
    )

    results = query.all()

    # Convert to list of dicts
    metrics = []
    for row in results:
        metrics.append(
            {
                "run_name": row.run_name,
                "model_name": row.model_name,
                "model_hint": row.model_hint,
                "total_latency_ms": (
                    float(row.total_latency_ms) if row.total_latency_ms else 0.0
                ),
                "avg_latency_ms": (
                    float(row.avg_latency_ms) if row.avg_latency_ms else 0.0
                ),
                "total_cost_usd": (
                    float(row.total_cost_usd) if row.total_cost_usd else 0.0
                ),
                "avg_cost_usd": float(row.avg_cost_usd) if row.avg_cost_usd else 0.0,
                "num_judgements": int(row.num_judgements),
            }
        )

    return metrics


def main():
    """Main execution."""
    # Load environment configuration
    load_runtime_config()

    # Create database engine
    engine = get_engine()

    print("Extracting cost and latency metrics from database...")

    with session_context(engine) as session:
        metrics = extract_run_metrics(session)

    print(f"Extracted metrics for {len(metrics)} runs")

    # Save to JSON
    output_path = Path(__file__).parent.parent.parent / OUTPUT_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nData saved to: {output_path}")
    print(f"\nSummary:")
    for metric in metrics:
        print(
            f"  {metric['run_name']} ({metric['model_hint']}): "
            f"${metric['total_cost_usd']:.4f}, "
            f"{metric['total_latency_ms']/1000:.2f}s, "
            f"{metric['num_judgements']} judgements"
        )


if __name__ == "__main__":
    main()
