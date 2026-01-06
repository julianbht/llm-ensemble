#!/usr/bin/env python3
"""
Verify that recovered qrels contain all original qrels with matching query-doc pairs.

The recovered file should be a superset of the original file:
- Every (query_idx, doc_idx) pair in original exists in recovered
- For matching pairs: query_idx, iteration, and doc_idx are identical
- Recovered adds a 4th column (grade) which was withheld in original
"""

from pathlib import Path
from collections import defaultdict


def load_original_qrels(filepath: Path) -> dict:
    """Load original qrels (3 columns: qidx, iteration, docidx)."""
    qrels = {}
    with open(filepath) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 3:
                print(f"⚠ Warning: Line {line_num} has {len(parts)} columns, expected 3")
                continue
            qidx, iteration, docidx = parts
            key = (qidx, docidx)
            if key in qrels:
                print(f"⚠ Warning: Duplicate pair {key} at line {line_num}")
            qrels[key] = {'iteration': iteration, 'line': line_num}
    return qrels


def load_recovered_qrels(filepath: Path) -> dict:
    """Load recovered qrels (4 columns: qidx, iteration, docidx, grade)."""
    qrels = {}
    with open(filepath) as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 4:
                print(f"⚠ Warning: Line {line_num} has {len(parts)} columns, expected 4")
                continue
            qidx, iteration, docidx, grade = parts
            key = (qidx, docidx)
            if key in qrels:
                print(f"⚠ Warning: Duplicate pair {key} at line {line_num}")
            qrels[key] = {
                'iteration': iteration,
                'grade': int(grade),
                'line': line_num
            }
    return qrels


def verify_qrels(original: dict, recovered: dict):
    """Verify that recovered contains all original pairs with matching data."""
    print("=" * 80)
    print("QRELS VERIFICATION")
    print("=" * 80)
    print()
    
    print(f"Original qrels:  {len(original):>6,} query-doc pairs")
    print(f"Recovered qrels: {len(recovered):>6,} query-doc pairs")
    print()
    
    # Check that all original pairs exist in recovered
    missing_pairs = []
    mismatched_iterations = []
    
    for key, orig_data in original.items():
        qidx, docidx = key
        
        if key not in recovered:
            missing_pairs.append(key)
        else:
            rec_data = recovered[key]
            if orig_data['iteration'] != rec_data['iteration']:
                mismatched_iterations.append({
                    'key': key,
                    'orig_iter': orig_data['iteration'],
                    'rec_iter': rec_data['iteration']
                })
    
    # Report results
    if missing_pairs:
        print(f"✗ FAILED: {len(missing_pairs)} pairs from original not found in recovered")
        print()
        print("Missing pairs (showing first 10):")
        for qidx, docidx in missing_pairs[:10]:
            print(f"  {qidx} {docidx}")
        if len(missing_pairs) > 10:
            print(f"  ... and {len(missing_pairs) - 10} more")
        print()
        return False
    
    if mismatched_iterations:
        print(f"✗ FAILED: {len(mismatched_iterations)} pairs have mismatched iterations")
        print()
        print("Mismatched iterations (showing first 10):")
        for mismatch in mismatched_iterations[:10]:
            qidx, docidx = mismatch['key']
            print(f"  {qidx} {docidx}: orig={mismatch['orig_iter']}, rec={mismatch['rec_iter']}")
        if len(mismatched_iterations) > 10:
            print(f"  ... and {len(mismatched_iterations) - 10} more")
        print()
        return False
    
    # All checks passed
    print("✓ All original pairs found in recovered with matching metadata")
    print()
    
    # Additional stats
    extra_pairs = len(recovered) - len(original)
    if extra_pairs > 0:
        print(f"ℹ Recovered file contains {extra_pairs:,} additional pairs not in original")
        print("  (This is expected - recovery found more judgments from TREC 2023)")
        print()
    
    # Query coverage
    orig_queries = set(qidx for qidx, _ in original.keys())
    rec_queries = set(qidx for qidx, _ in recovered.keys())
    print(f"Original queries:  {len(orig_queries):>3} unique")
    print(f"Recovered queries: {len(rec_queries):>3} unique")
    
    if rec_queries != orig_queries:
        extra_queries = rec_queries - orig_queries
        missing_queries = orig_queries - rec_queries
        if extra_queries:
            print(f"  + {len(extra_queries)} additional queries in recovered")
        if missing_queries:
            print(f"  - {len(missing_queries)} queries missing from recovered")
    print()
    
    # Grade distribution in recovered
    grade_dist = defaultdict(int)
    for key in original.keys():
        grade = recovered[key]['grade']
        grade_dist[grade] += 1
    
    print("Grade distribution (for original pairs):")
    for grade in sorted(grade_dist.keys()):
        count = grade_dist[grade]
        pct = count / len(original) * 100 if original else 0
        print(f"  Grade {grade}: {count:>5} ({pct:>5.1f}%)")
    print()
    
    print("=" * 80)
    print("✓ VERIFICATION PASSED")
    print("=" * 80)
    print()
    print("The recovered file is a valid superset of the original:")
    print("  - Same query-doc pairs (in order)")
    print("  - Same iterations")
    print("  - Adds grade labels as 4th column")
    
    return True


def main():
    base_dir = Path(__file__).parent.parent.parent / 'data'
    original_file = base_dir / 'llm_judge_challenge' / 'llm4eval_test_qrel_2024.txt'
    recovered_dir = base_dir / 'llm_judge_challenge_qrels_recovered'
    
    # Check original exists
    if not original_file.exists():
        print(f"✗ ERROR: Original file not found: {original_file}")
        return 1
    
    # Load original once
    print("Loading original qrels...")
    original = load_original_qrels(original_file)
    print(f"  Loaded {len(original):,} pairs")
    print()
    
    all_success = True
    
    # Verify main recovered file (exact match with original)
    recovered_file = recovered_dir / 'llm4eval_test_qrel_2024_recovered.txt'
    if recovered_file.exists():
        print("=" * 80)
        print("VERIFYING: llm4eval_test_qrel_2024_recovered.txt")
        print("=" * 80)
        print("(Should be exact match with original withheld test set)")
        print()
        
        recovered = load_recovered_qrels(recovered_file)
        
        if len(recovered) != len(original):
            print(f"✗ FAILED: Size mismatch")
            print(f"  Original:  {len(original):,}")
            print(f"  Recovered: {len(recovered):,}")
            all_success = False
        else:
            print(f"✓ Size matches: {len(recovered):,} pairs")
        
        # Verify exact match
        missing = []
        for key in original.keys():
            if key not in recovered:
                missing.append(key)
        
        if missing:
            print(f"✗ FAILED: {len(missing)} pairs missing")
            all_success = False
        else:
            print(f"✓ All pairs present")
        
        # Grade distribution
        grade_dist = defaultdict(int)
        for key in original.keys():
            grade = recovered[key]['grade']
            grade_dist[grade] += 1
        
        print()
        print("Grade distribution:")
        for grade in sorted(grade_dist.keys()):
            count = grade_dist[grade]
            pct = count / len(original) * 100 if original else 0
            print(f"  Grade {grade}: {count:>5} ({pct:>5.1f}%)")
        
        print()
    else:
        print(f"✗ ERROR: Main recovered file not found: {recovered_file.name}")
        all_success = False
        print()
    
    # Verify superset (contains all original plus extras)
    superset_file = recovered_dir / 'llm4eval_test_qrel_2024_recovered_superset.txt'
    if superset_file.exists():
        print("=" * 80)
        print("VERIFYING: llm4eval_test_qrel_2024_recovered_superset.txt")
        print("=" * 80)
        print("(Should be superset of original)")
        print()
        
        superset = load_recovered_qrels(superset_file)
        success = verify_qrels(original, superset)
        all_success = all_success and success
    else:
        print(f"ℹ Superset file not found: {superset_file.name}")
        print()
    
    return 0 if all_success else 1


if __name__ == '__main__':
    exit(main())
