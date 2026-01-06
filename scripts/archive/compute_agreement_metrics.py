#!/usr/bin/env python3
"""
Compute agreement metrics between LLM submission and official TREC qrels.
Validates that we have the correct ground truth data.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import kendalltau, spearmanr


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
    """Load submission file and convert indices to original IDs."""
    predictions = {}  # (qid, doc_id) -> predicted_grade

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qidx, _, docidx, grade = parts

                # Convert indices to original IDs
                if qidx in qid_map and docidx in docid_map:
                    qid = qid_map[qidx]
                    doc_id = docid_map[docidx]
                    predictions[(qid, doc_id)] = int(grade)

    return predictions


def load_qrels(filepath: Path) -> dict:
    """Load official qrels file."""
    qrels = {}  # (qid, doc_id) -> gold_grade

    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, _, doc_id, grade = parts
                qrels[(qid, doc_id)] = int(grade)

    return qrels


def compute_krippendorff_alpha(y_pred, y_true):
    """Compute Krippendorff's alpha using the definition."""
    try:
        import krippendorff
        # krippendorff expects data in format: (n_annotators, n_items)
        # We have 2 annotators (prediction, gold) and n_items samples
        data = np.array([y_pred, y_true])
        alpha = krippendorff.alpha(reliability_data=data, level_of_measurement='ordinal')
        return alpha
    except ImportError:
        print("  ⚠ krippendorff package not installed. Install with: pip install krippendorff")
        return None


def main():
    data_dir = Path(__file__).parent.parent / 'data'

    print("=" * 80)
    print("AGREEMENT METRICS COMPUTATION")
    print("=" * 80)
    print()

    # Load mappings
    print("Loading ID mappings...")
    qid_map = load_mapping(data_dir / 'qid_to_qidx.txt', reverse=True)
    docid_map = load_mapping(data_dir / 'docid_to_docidx.txt', reverse=True)
    print(f"  Query mapping: {len(qid_map)} queries")
    print(f"  Doc mapping: {len(docid_map)} documents")
    print()

    # Load submission
    print("Loading submission predictions...")
    submission_file = data_dir / 'NISTRetrieval-instruct0.txt'
    predictions = load_submission(submission_file, qid_map, docid_map)
    print(f"  Loaded {len(predictions)} predictions")
    print()

    # Load official qrels
    print("Loading official qrels...")
    qrels_file = data_dir / 'llm4eval_official_qrels_2023.txt'
    qrels = load_qrels(qrels_file)
    print(f"  Loaded {len(qrels)} gold judgments")
    print()

    # Align predictions with gold labels
    print("Aligning predictions with gold labels...")
    aligned_pairs = []
    y_pred = []
    y_true = []

    for (qid, doc_id), pred_grade in predictions.items():
        if (qid, doc_id) in qrels:
            gold_grade = qrels[(qid, doc_id)]
            aligned_pairs.append((qid, doc_id, pred_grade, gold_grade))
            y_pred.append(pred_grade)
            y_true.append(gold_grade)

    print(f"  Aligned {len(aligned_pairs)} query-document pairs")
    print()

    if len(aligned_pairs) == 0:
        print("ERROR: No aligned pairs found!")
        print("This suggests ID mappings are incorrect.")
        return

    # Convert to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Show sample alignments
    print("Sample alignments:")
    for qid, doc_id, pred, gold in aligned_pairs[:5]:
        match = "✓" if pred == gold else "✗"
        print(f"  {match} Q:{qid[:10]:>10}... D:{doc_id[:20]:>20}... Pred:{pred} Gold:{gold}")
    print()

    # Grade distribution comparison
    print("Grade distribution:")
    print("  Grade | Predicted | Gold")
    print("  ------|-----------|-----")
    for grade in range(4):
        pred_count = np.sum(y_pred == grade)
        gold_count = np.sum(y_true == grade)
        print(f"    {grade}   | {pred_count:>9} | {gold_count:>4}")
    print()

    # Compute agreement metrics
    print("=" * 80)
    print("AGREEMENT METRICS")
    print("=" * 80)
    print()

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's κ (Kappa):              {kappa:.4f}")

    # Kendall's Tau (instance-level)
    tau_instance, tau_pval_instance = kendalltau(y_true, y_pred)
    print(f"Kendall's τ (instance-level):   {tau_instance:.4f} (p-value: {tau_pval_instance:.4e})")

    # Spearman's Rho (instance-level)
    rho_instance, rho_pval_instance = spearmanr(y_true, y_pred)
    print(f"Spearman's ρ (instance-level):  {rho_instance:.4f} (p-value: {rho_pval_instance:.4e})")

    print()
    print("Computing ranking correlations per query...")

    # Group by query for ranking correlation
    query_groups = defaultdict(lambda: {'pred': [], 'gold': []})
    for qid, doc_id, pred, gold in aligned_pairs:
        query_groups[qid]['pred'].append(pred)
        query_groups[qid]['gold'].append(gold)

    # Compute per-query correlations
    tau_per_query = []
    rho_per_query = []

    for qid, data in query_groups.items():
        pred_ranks = data['pred']
        gold_ranks = data['gold']

        if len(pred_ranks) > 1:  # Need at least 2 items for correlation
            try:
                tau_q, _ = kendalltau(gold_ranks, pred_ranks)
                rho_q, _ = spearmanr(gold_ranks, pred_ranks)

                # Filter out NaN values (can happen with constant rankings)
                if not np.isnan(tau_q):
                    tau_per_query.append(tau_q)
                if not np.isnan(rho_q):
                    rho_per_query.append(rho_q)
            except:
                pass

    # Average per-query correlations
    tau_avg = np.mean(tau_per_query) if tau_per_query else 0.0
    rho_avg = np.mean(rho_per_query) if rho_per_query else 0.0

    print(f"Kendall's τ (per-query avg):    {tau_avg:.4f} (computed over {len(tau_per_query)} queries)")
    print(f"Spearman's ρ (per-query avg):   {rho_avg:.4f} (computed over {len(rho_per_query)} queries)")

    # Krippendorff's Alpha
    alpha = compute_krippendorff_alpha(y_pred.tolist(), y_true.tolist())
    if alpha is not None:
        print(f"Krippendorff's α (Alpha):       {alpha:.4f}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total aligned pairs: {len(aligned_pairs)}")
    print(f"Exact agreement: {np.sum(y_pred == y_true)} / {len(y_pred)} ({np.sum(y_pred == y_true) / len(y_pred) * 100:.1f}%)")
    print()
    print("These metrics should match the paper if we have the correct data.")


if __name__ == '__main__':
    main()
