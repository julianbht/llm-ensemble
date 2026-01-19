#!/usr/bin/env python3
"""Compute binarized classification metrics for different relevance thresholds.

Computes precision, recall, accuracy, and F1 for three binarization schemes:
- 0|123: 0 = irrelevant, 1,2,3 = relevant
- 01|23: 0,1 = irrelevant, 2,3 = relevant
- 012|3: 0,1,2 = irrelevant, 3 = relevant

Outputs a combined table showing all metrics across thresholds.

Usage:
    python scripts/analytics/compute_binarized_metrics.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Load environment variables
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import evaluate startup to ensure all ORMs are registered
from llm_ensemble.evaluate.startup import dependency_configurator  # noqa: F401
from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory


# ============================================================================
# CONFIGURATION
# ============================================================================

# Runs to analyze (run_name, run_type, display_label)
RUNS: List[Tuple[str, str, str]] = [
    ("reference-ensemble-gpt-5-1-all-samples-start", "infer", "GPT-5.1"),
    ("-ensemble-1-to-5-majority-vote-average", "aggregate", "Ensemble (MVA)"),
]

# Binarization thresholds: (name, irrelevant_classes, relevant_classes)
THRESHOLDS = [
    ("0|123", [0], [1, 2, 3]),
    ("01|23", [0, 1], [2, 3]),
    ("012|3", [0, 1, 2], [3]),
]

# ============================================================================


@dataclass
class BinarizedMetrics:
    """Metrics for a single binarization threshold."""

    threshold_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_positive_true: int  # number of relevant in ground truth
    n_positive_pred: int  # number of relevant in predictions


def load_evaluation_data(run_name: str, run_type: str) -> Tuple[List[int], List[int]]:
    """Load ground truth and predictions from a run."""
    io_name = "db_aggregate_to_json" if run_type == "aggregate" else "db_infer_to_json"
    reader = IOAdapterFactory.create_reader(io_name)
    eval_data = reader.read(run_name)

    ground_truth = [int(gt) for gt in eval_data.ground_truth]
    predictions = [
        int(pred) if pred is not None else -1 for pred in eval_data.predictions
    ]

    return ground_truth, predictions


def binarize(
    values: List[int], irrelevant_classes: List[int], relevant_classes: List[int]
) -> List[int]:
    """Binarize values: 0 for irrelevant, 1 for relevant, -1 for invalid."""
    result = []
    for v in values:
        if v in irrelevant_classes:
            result.append(0)
        elif v in relevant_classes:
            result.append(1)
        else:
            result.append(-1)  # invalid (e.g., unparseable)
    return result


def compute_binarized_metrics(
    ground_truth: List[int],
    predictions: List[int],
    threshold_name: str,
    irrelevant_classes: List[int],
    relevant_classes: List[int],
) -> BinarizedMetrics:
    """Compute metrics for a single binarization threshold."""
    # Binarize both
    gt_binary = binarize(ground_truth, irrelevant_classes, relevant_classes)
    pred_binary = binarize(predictions, irrelevant_classes, relevant_classes)

    # Filter out invalid predictions
    valid_indices = [i for i, p in enumerate(pred_binary) if p >= 0]
    gt_valid = [gt_binary[i] for i in valid_indices]
    pred_valid = [pred_binary[i] for i in valid_indices]

    # Compute metrics (positive class = 1 = relevant)
    accuracy = accuracy_score(gt_valid, pred_valid)
    precision = precision_score(gt_valid, pred_valid, zero_division=0)
    recall = recall_score(gt_valid, pred_valid, zero_division=0)
    f1 = f1_score(gt_valid, pred_valid, zero_division=0)

    n_positive_true = sum(gt_valid)
    n_positive_pred = sum(pred_valid)

    return BinarizedMetrics(
        threshold_name=threshold_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        n_positive_true=n_positive_true,
        n_positive_pred=n_positive_pred,
    )


def print_metrics_table(
    run_label: str, metrics_list: List[BinarizedMetrics], total_samples: int
):
    """Print metrics as a formatted table."""
    print(f"\n{'=' * 70}")
    print(f"BINARIZED METRICS: {run_label}")
    print(f"{'=' * 70}")
    print(f"Total samples: {total_samples:,}")
    print()
    print(
        f"{'Threshold':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'#Pos(GT)':>10} {'#Pos(Pred)':>12}"
    )
    print("-" * 76)

    for m in metrics_list:
        print(
            f"{m.threshold_name:<12} "
            f"{m.accuracy:>10.1%} "
            f"{m.precision:>10.1%} "
            f"{m.recall:>10.1%} "
            f"{m.f1:>10.3f} "
            f"{m.n_positive_true:>10,} "
            f"{m.n_positive_pred:>12,}"
        )


def save_latex_table(
    all_results: List[Tuple[str, List[BinarizedMetrics]]], output_path: Path
):
    """Save all results as a LaTeX table."""
    lines = []
    lines.append("% Binarized Classification Metrics")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Binary classification metrics at different relevance thresholds}"
    )
    lines.append("\\label{tab:binarized-metrics}")
    lines.append("\\begin{tabular}{ll|rrrr}")
    lines.append("\\toprule")
    lines.append("Model & Threshold & Accuracy & Precision & Recall & F1 \\\\")
    lines.append("\\midrule")

    for run_label, metrics_list in all_results:
        for i, m in enumerate(metrics_list):
            label = run_label if i == 0 else ""
            lines.append(
                f"{label} & {m.threshold_name} & "
                f"{m.accuracy:.1%} & {m.precision:.1%} & {m.recall:.1%} & {m.f1:.3f} \\\\".replace(
                    "%", "\\%"
                )
            )
        lines.append("\\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nLaTeX table saved to: {output_path}")


def main():
    """Main execution."""
    project_root = Path(__file__).parent.parent.parent
    tables_dir = project_root / "artifacts" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for run_name, run_type, run_label in RUNS:
        print(f"\nLoading {run_label} ({run_name})...")

        try:
            ground_truth, predictions = load_evaluation_data(run_name, run_type)
        except Exception as e:
            print(f"  Error loading {run_name}: {e}")
            continue

        # Filter valid predictions for sample count
        valid_count = sum(1 for p in predictions if p >= 0)
        print(
            f"  Loaded {len(ground_truth):,} samples ({valid_count:,} valid predictions)"
        )

        metrics_list = []
        for threshold_name, irrelevant, relevant in THRESHOLDS:
            metrics = compute_binarized_metrics(
                ground_truth, predictions, threshold_name, irrelevant, relevant
            )
            metrics_list.append(metrics)

        print_metrics_table(run_label, metrics_list, valid_count)
        all_results.append((run_label, metrics_list))

    # Save combined LaTeX table
    if all_results:
        latex_path = tables_dir / "binarized_metrics.tex"
        save_latex_table(all_results, latex_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
