#!/usr/bin/env python3
"""
Step 5: Convert TREC qrels to challenge format with anonymized indices.

Takes TREC format qrels (real IDs) and converts to challenge format
(anonymized q/p indices) for use with llm4eval_query_2024.txt and
llm4eval_document_2024.jsonl.

Output is a drop-in replacement for llm4eval_test_qrel_2024.txt with
actual grades instead of zeros.
"""

from pathlib import Path
from collections import defaultdict


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


def main():
    base_dir = Path(__file__).parent.parent.parent / 'data'
    challenge_dir = base_dir / 'llm_judge_challenge'
    recovered_dir = base_dir / 'llm_judge_challenge_qrels_recovered'

    print("=" * 80)
    print("STEP 5: CONVERT TO CHALLENGE FORMAT")
    print("=" * 80)
    print()

    # Load ID mappings
    print("Loading ID mappings...")
    qid_to_idx = load_mapping(challenge_dir / 'qid_to_qidx.txt')
    docid_to_idx = load_mapping(challenge_dir / 'docid_to_docidx.txt')
    print(f"  Query mapping: {len(qid_to_idx)} IDs")
    print(f"  Doc mapping: {len(docid_to_idx)} IDs")
    print()

    # Load TREC format qrels
    print("Loading TREC format qrels...")
    trec_qrels_file = recovered_dir / 'trec_2023_challenge_subset.txt'

    if not trec_qrels_file.exists():
        print(f"✗ ERROR: {trec_qrels_file} not found")
        print("  Run step3_extract_challenge_qrels.py first")
        return 1

    challenge_qrels = []
    unmapped_queries = set()
    unmapped_docs = set()
    total_read = 0

    with open(trec_qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                qid, iteration, doc_id, grade = parts
                total_read += 1

                # Convert to anonymized indices
                if qid in qid_to_idx and doc_id in docid_to_idx:
                    qidx = qid_to_idx[qid]
                    docidx = docid_to_idx[doc_id]
                    challenge_qrels.append((qidx, iteration, docidx, int(grade)))
                else:
                    if qid not in qid_to_idx:
                        unmapped_queries.add(qid)
                    if doc_id not in docid_to_idx:
                        unmapped_docs.add(doc_id)

    print(f"  Read {total_read:,} qrels from TREC format")
    print(f"  Converted {len(challenge_qrels):,} qrels to challenge format")
    print()

    if unmapped_queries or unmapped_docs:
        print("⚠ Warning: Some IDs could not be mapped:")
        if unmapped_queries:
            print(f"  Unmapped queries: {len(unmapped_queries)}")
        if unmapped_docs:
            print(f"  Unmapped docs: {len(unmapped_docs)}")
        print()

    # Grade distribution
    grade_dist = defaultdict(int)
    for _, _, _, grade in challenge_qrels:
        grade_dist[grade] += 1

    print("Grade distribution:")
    for grade in sorted(grade_dist.keys()):
        count = grade_dist[grade]
        pct = count / len(challenge_qrels) * 100 if challenge_qrels else 0
        print(f"  Grade {grade}: {count:>5} ({pct:>5.1f}%)")
    print()

    # First create the main recovered file (original withheld pairs only)
    print("Creating recovered qrels (original withheld pairs only)...")
    original_withheld = challenge_dir / 'llm4eval_test_qrel_2024.txt'
    
    if original_withheld.exists():
        # Load original pairs to preserve order
        original_pairs = []
        with open(original_withheld) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    original_pairs.append((parts[0], parts[1], parts[2]))
        
        # Filter recovered qrels to only original pairs
        challenge_qrels_dict = {(qidx, docidx): (iteration, grade) 
                                for qidx, iteration, docidx, grade in challenge_qrels}
        
        recovered_qrels = []
        for qidx, iteration, docidx in original_pairs:
            key = (qidx, docidx)
            if key in challenge_qrels_dict:
                _, grade = challenge_qrels_dict[key]
                recovered_qrels.append(f"{qidx} {iteration} {docidx} {grade}\n")
        
        output_file = recovered_dir / 'llm4eval_test_qrel_2024_recovered.txt'
        with open(output_file, 'w') as f:
            f.writelines(recovered_qrels)
        
        print(f"✓ Recovered qrels written to: {output_file.name}")
        print(f"  Total judgments: {len(recovered_qrels):,}")
        print(f"  (Matches original withheld test set)")
    else:
        print(f"⚠ Warning: {original_withheld} not found")
        print(f"  Cannot create filtered recovered file")
    
    print()

    # Also write superset with all recovered judgments
    superset_file = recovered_dir / 'llm4eval_test_qrel_2024_recovered_superset.txt'
    with open(superset_file, 'w') as f:
        for qidx, iteration, docidx, grade in sorted(challenge_qrels):
            f.write(f"{qidx} {iteration} {docidx} {grade}\n")

    print(f"✓ Recovered superset written to: {superset_file.name}")
    print(f"  Total judgments: {len(challenge_qrels):,}")
    print(f"  (Contains additional queries from TREC 2023)")
    
    print()

    # Show sample
    print("Sample qrels (challenge format):")
    for qidx, iteration, docidx, grade in sorted(challenge_qrels)[:5]:
        print(f"  {qidx} {iteration} {docidx} {grade}")

    print()
    print("=" * 80)
    print("USAGE")
    print("=" * 80)
    print()
    print("Two files created:")
    print()
    print("1. llm4eval_test_qrel_2024_recovered.txt (MAIN FILE)")
    print("   - 4,423 judgments for 25 queries")
    print("   - Exact same pairs as original withheld test set")
    print("   - **Use this for fair comparison with LLM Judge Challenge**")
    print()
    print("2. llm4eval_test_qrel_2024_recovered_superset.txt (OPTIONAL)")
    print("   - 11,718 judgments for 50 queries")
    print("   - Contains additional queries from TREC 2023")
    print("   - Use if you want more test data")
    print()
    print("Both use with:")
    print("  - llm4eval_query_2024.txt")
    print("  - llm4eval_document_2024.jsonl")
    print()
    print("Format: query_idx iteration doc_idx grade")

    return 0


if __name__ == '__main__':
    exit(main())
