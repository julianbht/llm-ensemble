#!/usr/bin/env python3
"""
Extract official TREC 2023 qrels for the 50 queries in the LLM4Eval challenge.
"""

from pathlib import Path
from collections import defaultdict


def load_query_mapping(filepath: Path) -> set:
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
    data_dir = Path(__file__).parent.parent / 'data'

    # Load the 50 challenge query IDs
    print("Loading challenge query IDs...")
    challenge_qids = load_query_mapping(data_dir / 'qid_to_qidx.txt')
    print(f"Found {len(challenge_qids)} queries in challenge")
    print()

    # Load official TREC 2023 qrels
    print("Loading official TREC 2023 qrels...")
    qrels_file = data_dir / 'trec_2023_passage_qrels_official.txt'

    if not qrels_file.exists():
        print(f"Error: {qrels_file} not found")
        print("Run verify_trec_dl_2023.py first to download qrels")
        return

    # Parse qrels and filter for challenge queries
    challenge_qrels = []
    stats = defaultdict(int)
    grade_distribution = defaultdict(int)

    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, iteration, doc_id, grade = parts
                grade = int(grade)

                stats['total_qrels'] += 1

                if qid in challenge_qids:
                    challenge_qrels.append((qid, iteration, doc_id, grade))
                    stats['challenge_qrels'] += 1
                    grade_distribution[grade] += 1

    print(f"Total qrels in TREC 2023: {stats['total_qrels']:,}")
    print(f"Qrels for challenge queries: {stats['challenge_qrels']:,}")
    print()

    print("Relevance grade distribution:")
    for grade in sorted(grade_distribution.keys()):
        count = grade_distribution[grade]
        pct = (count / stats['challenge_qrels'] * 100) if stats['challenge_qrels'] > 0 else 0
        print(f"  Grade {grade}: {count:>5} ({pct:>5.1f}%)")
    print()

    # Count queries with judgments
    queries_with_qrels = set(qid for qid, _, _, _ in challenge_qrels)
    print(f"Challenge queries with qrels: {len(queries_with_qrels)} / {len(challenge_qids)}")

    missing_qids = challenge_qids - queries_with_qrels
    if missing_qids:
        print(f"\n⚠ Missing qrels for {len(missing_qids)} queries:")
        for qid in sorted(missing_qids)[:10]:
            print(f"  {qid}")
        if len(missing_qids) > 10:
            print(f"  ... and {len(missing_qids) - 10} more")
    else:
        print("✓ All challenge queries have qrels!")

    # Write filtered qrels
    output_file = data_dir / 'llm4eval_official_qrels_2023.txt'
    with open(output_file, 'w') as f:
        for qid, iteration, doc_id, grade in sorted(challenge_qrels):
            f.write(f"{qid} {iteration} {doc_id} {grade}\n")

    print()
    print(f"✓ Extracted qrels written to: {output_file}")
    print()

    # Show sample
    print("Sample qrels:")
    sample_qid = list(queries_with_qrels)[0]
    sample_qrels = [(q, d, g) for q, _, d, g in challenge_qrels if q == sample_qid][:5]
    print(f"\nQuery {sample_qid}:")
    for qid, doc_id, grade in sample_qrels:
        print(f"  {doc_id} -> Grade {grade}")


if __name__ == '__main__':
    main()
