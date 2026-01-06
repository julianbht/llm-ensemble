#!/usr/bin/env python3
"""
Step 1: Verify that challenge queries come from TREC 2023 Deep Learning Track.

Loads query IDs from mapping file and matches them against TREC 2023 DL queries
using ir_datasets library.
"""

from pathlib import Path


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
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'llm_judge_challenge'

    print("=" * 80)
    print("STEP 1: VERIFY DATA SOURCE")
    print("=" * 80)
    print()

    # Load challenge query IDs
    print("Loading challenge query IDs...")
    qid_file = data_dir / 'qid_to_qidx.txt'
    challenge_qids = load_query_ids(qid_file)
    print(f"  Found {len(challenge_qids)} queries")
    print(f"  Sample IDs: {list(challenge_qids)[:5]}")
    print()

    # Match against TREC 2023 DL using ir_datasets
    try:
        import ir_datasets

        print("Loading TREC 2023 DL Track from ir_datasets...")
        dataset = ir_datasets.load('msmarco-passage-v2/trec-dl-2023')
        print("  ✓ Dataset loaded successfully")
        print()

        # Extract all TREC 2023 query IDs
        print("Extracting TREC 2023 query IDs...")
        trec_qids = set()
        for query in dataset.queries_iter():
            trec_qids.add(query.query_id)

        print(f"  TREC 2023 has {len(trec_qids)} total queries")
        print()

        # Find matches
        matching = challenge_qids & trec_qids
        match_rate = len(matching) / len(challenge_qids) * 100

        print("=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Challenge queries:     {len(challenge_qids)}")
        print(f"Matching TREC queries: {len(matching)}")
        print(f"Match rate:            {match_rate:.1f}%")
        print()

        if match_rate == 100:
            print("✓ VERIFIED: All challenge queries come from TREC 2023 DL Track")
        elif match_rate > 90:
            print("⚠ PARTIAL MATCH: Most queries match, but some missing")
            missing = challenge_qids - trec_qids
            print(f"  Missing: {missing}")
        else:
            print("✗ LOW MATCH: Queries likely from different source")

    except ImportError:
        print("ERROR: ir_datasets not installed")
        print("Install with: pip install ir-datasets")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
