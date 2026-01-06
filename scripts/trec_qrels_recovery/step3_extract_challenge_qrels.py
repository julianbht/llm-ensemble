#!/usr/bin/env python3
"""
Step 3: Extract qrels for the 50 challenge queries from full TREC 2023 qrels.

Filters the full TREC 2023 qrels (22K+ judgments) to only include the 50 queries
used in the LLM4Eval challenge.
"""

from pathlib import Path
from collections import defaultdict


def load_query_ids(filepath: Path) -> set:
    """Load query IDs from mapping file."""
    query_ids = set()
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                qid, _ = parts
                query_ids.add(qid)
    return query_ids


def main():
    base_dir = Path(__file__).parent.parent.parent / 'data'
    challenge_dir = base_dir / 'llm_judge_challenge'
    test_dir = base_dir / 'llm_judge_challenge_qrels_recovered'

    print("=" * 80)
    print("STEP 3: EXTRACT CHALLENGE QRELS")
    print("=" * 80)
    print()

    # Load challenge query IDs
    print("Loading challenge query IDs...")
    qid_file = challenge_dir / 'qid_to_qidx.txt'
    challenge_qids = load_query_ids(qid_file)
    print(f"  Found {len(challenge_qids)} queries")
    print()

    # Load full TREC qrels
    print("Loading full TREC 2023 qrels...")
    qrels_file = test_dir / 'trec_2023_passage_qrels_official.txt'

    if not qrels_file.exists():
        print(f"✗ ERROR: {qrels_file} not found")
        print("  Run step2_download_trec_qrels.py first")
        return 1

    # Parse and filter qrels
    challenge_qrels = []
    grade_dist = defaultdict(int)
    total_qrels = 0

    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, iteration, doc_id, grade = parts
                grade = int(grade)
                total_qrels += 1

                if qid in challenge_qids:
                    challenge_qrels.append((qid, iteration, doc_id, grade))
                    grade_dist[grade] += 1

    print(f"  Total TREC qrels: {total_qrels:,}")
    print(f"  Challenge qrels:  {len(challenge_qrels):,}")
    print()

    # Grade distribution
    print("Grade distribution:")
    for grade in sorted(grade_dist.keys()):
        count = grade_dist[grade]
        pct = count / len(challenge_qrels) * 100
        print(f"  Grade {grade}: {count:>5} ({pct:>5.1f}%)")
    print()

    # Check coverage
    queries_with_qrels = set(qid for qid, _, _, _ in challenge_qrels)
    print(f"Queries with qrels: {len(queries_with_qrels)} / {len(challenge_qids)}")

    missing = challenge_qids - queries_with_qrels
    if missing:
        print(f"  ⚠ Missing qrels for {len(missing)} queries:")
        for qid in sorted(missing)[:5]:
            print(f"    {qid}")
    else:
        print("  ✓ All challenge queries have qrels")

    print()

    # Write filtered qrels
    output_file = test_dir / 'trec_2023_challenge_subset.txt'
    with open(output_file, 'w') as f:
        for qid, iteration, doc_id, grade in sorted(challenge_qrels):
            f.write(f"{qid} {iteration} {doc_id} {grade}\n")

    print(f"✓ Challenge subset qrels written to: {output_file.name}")
    print(f"  Total judgments: {len(challenge_qrels):,}")
    print(f"  Format: TREC (real IDs, not anonymized)")

    return 0


if __name__ == '__main__':
    exit(main())
