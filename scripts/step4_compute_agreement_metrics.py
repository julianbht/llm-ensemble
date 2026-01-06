#!/usr/bin/env python3
"""
Step 4: Compute agreement metrics between LLM submission and gold labels.

Computes Cohen's Kappa and Krippendorff's Alpha to validate that we have
the correct gold labels.
"""

from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score


def load_mapping(filepath: Path, reverse=False) -> dict:
    """Load ID mapping file."""
    mapping = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                original_id, idx = parts
                if reverse:
                    mapping[idx] = original_id
                else:
                    mapping[original_id] = idx
    return mapping


def load_submission(filepath: Path, qid_map: dict, docid_map: dict) -> dict:
    """Load submission and map indices to original IDs."""
    predictions = {}

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qidx, _, docidx, grade = parts

                if qidx in qid_map and docidx in docid_map:
                    qid = qid_map[qidx]
                    doc_id = docid_map[docidx]
                    predictions[(qid, doc_id)] = int(grade)

    return predictions


def load_qrels(filepath: Path) -> dict:
    """Load official qrels."""
    qrels = {}

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, doc_id, grade = parts
                qrels[(qid, doc_id)] = int(grade)

    return qrels


def compute_krippendorff_alpha(y_pred, y_true):
    """Compute Krippendorff's alpha."""
    try:
        import krippendorff
        data = np.array([y_pred, y_true])
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='ordinal')
        return alpha
    except ImportError:
        print("  ⚠ krippendorff not installed: pip install krippendorff")
        return None


def main():
    data_dir = Path(__file__).parent.parent / 'data'

    print("=" * 80)
    print("STEP 4: COMPUTE AGREEMENT METRICS")
    print("=" * 80)
    print()

    # Load mappings
    print("Loading ID mappings...")
    qid_map = load_mapping(data_dir / 'qid_to_qidx.txt', reverse=True)
    docid_map = load_mapping(data_dir / 'docid_to_docidx.txt', reverse=True)
    print(f"  Queries: {len(qid_map)}, Documents: {len(docid_map)}")
    print()

    # Load submission
    print("Loading submission...")
    submission_file = data_dir / 'NISTRetrieval-instruct0.txt'
    if not submission_file.exists():
        print(f"✗ ERROR: {submission_file} not found")
        return 1

    predictions = load_submission(submission_file, qid_map, docid_map)
    print(f"  Loaded {len(predictions)} predictions")
    print()

    # Load gold labels
    print("Loading gold labels...")
    qrels_file = data_dir / 'llm4eval_official_qrels_2023.txt'
    if not qrels_file.exists():
        print(f"✗ ERROR: {qrels_file} not found")
        print("  Run step3_extract_challenge_qrels.py first")
        return 1

    qrels = load_qrels(qrels_file)
    print(f"  Loaded {len(qrels)} gold judgments")
    print()

    # Align predictions with gold
    print("Aligning predictions with gold labels...")
    y_pred = []
    y_true = []

    for (qid, doc_id), pred_grade in predictions.items():
        if (qid, doc_id) in qrels:
            gold_grade = qrels[(qid, doc_id)]
            y_pred.append(pred_grade)
            y_true.append(gold_grade)

    print(f"  Aligned: {len(y_pred)} pairs")

    if len(y_pred) == 0:
        print("✗ ERROR: No aligned pairs found")
        return 1

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Grade distribution
    print()
    print("Grade distribution:")
    print("  Grade | Predicted | Gold")
    print("  ------|-----------|-----")
    for grade in range(4):
        pred_count = np.sum(y_pred == grade)
        gold_count = np.sum(y_true == grade)
        print(f"    {grade}   | {pred_count:>9} | {gold_count:>4}")

    # Agreement metrics
    print()
    print("=" * 80)
    print("AGREEMENT METRICS")
    print("=" * 80)

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's κ (Kappa):              {kappa:.4f}")

    alpha = compute_krippendorff_alpha(y_pred.tolist(), y_true.tolist())
    if alpha is not None:
        print(f"Krippendorff's α (Alpha):       {alpha:.4f}")

    # Summary
    exact_match = np.sum(y_pred == y_true)
    exact_pct = exact_match / len(y_pred) * 100

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Aligned pairs:   {len(y_pred):>6}")
    print(f"Exact agreement: {exact_match:>6} ({exact_pct:.1f}%)")
    print(f"Cohen's κ:       {kappa:>6.4f}")
    if alpha is not None:
        print(f"Krippendorff α:  {alpha:>6.4f}")

    print()
    print("✓ Agreement metrics computed successfully")

    return 0


if __name__ == '__main__':
    exit(main())
