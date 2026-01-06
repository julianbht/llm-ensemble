#!/usr/bin/env python3
"""
Verify if LLM4Eval challenge data comes from TREC 2023 Deep Learning track.
Matches query IDs and document IDs against ir_datasets to confirm source.
"""

import sys
from pathlib import Path
from typing import Dict, Set


def load_mapping(filepath: Path) -> Dict[str, str]:
    """Load ID mapping file (original_id -> idx)."""
    mapping = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                original_id, idx = parts
                mapping[original_id] = idx
    return mapping


def main():
    # Load mappings
    data_dir = Path(__file__).parent.parent / 'data'

    print("Loading mapping files...")
    docid_map = load_mapping(data_dir / 'docid_to_docidx.txt')
    qid_map = load_mapping(data_dir / 'qid_to_qidx.txt')

    print(f"Found {len(docid_map)} documents in mapping")
    print(f"Found {len(qid_map)} queries in mapping")
    print()

    # Extract actual MSMARCO passage IDs (strip the msmarco_passage_ prefix)
    # Format: msmarco_passage_XX_YYYYYY -> YYYYYY
    doc_ids = set()
    for doc_id in docid_map.keys():
        if doc_id.startswith('msmarco_passage_'):
            # Extract the actual passage ID (last part after final underscore)
            parts = doc_id.split('_')
            passage_id = parts[-1]
            doc_ids.add(passage_id)

    query_ids = set(qid_map.keys())

    print("Sample document IDs from mapping:")
    for i, doc_id in enumerate(list(doc_ids)[:5]):
        print(f"  {doc_id}")
    print()

    print("Sample query IDs from mapping:")
    for i, qid in enumerate(list(query_ids)[:5]):
        print(f"  {qid}")
    print()

    # Try to load TREC DL 2023 via ir_datasets
    print("Attempting to load TREC DL 2023 from ir_datasets...")
    try:
        import ir_datasets

        # TREC DL 2023 uses MS MARCO v2 passage corpus
        # Try different dataset identifiers
        dataset_candidates = [
            'msmarco-passage-v2/trec-dl-2023',
            'msmarco-passage/trec-dl-2023',
            'msmarco-v2-passage/trec-dl-2023',
        ]

        dataset = None
        for candidate in dataset_candidates:
            try:
                print(f"  Trying {candidate}...")
                dataset = ir_datasets.load(candidate)
                print(f"  ✓ Successfully loaded {candidate}")
                break
            except Exception as e:
                print(f"  ✗ {candidate} not found")

        if dataset is None:
            print("\nCouldn't find TREC DL 2023 dataset.")
            print("\nListing available TREC DL datasets...")
            available = [d for d in ir_datasets.registry._registered.keys() if 'trec-dl' in d.lower()]
            for d in sorted(available)[:10]:
                print(f"  - {d}")
            return

        print()
        print("=" * 80)
        print("MATCHING QUERY IDs")
        print("=" * 80)

        # Check queries
        trec_query_ids = set()
        print("Loading TREC DL 2023 queries...")
        for query in dataset.queries_iter():
            trec_query_ids.add(query.query_id)

        print(f"TREC DL 2023 has {len(trec_query_ids)} queries")

        # Find matches
        matching_queries = query_ids & trec_query_ids
        print(f"Matching queries: {len(matching_queries)} / {len(query_ids)}")

        if matching_queries:
            print("\nSample matching query IDs:")
            for qid in list(matching_queries)[:5]:
                print(f"  {qid}")

        print()
        print("=" * 80)
        print("CHECKING QRELS (Relevance Labels)")
        print("=" * 80)

        # Check what's available on the dataset
        print("Available dataset attributes:")
        attrs = [attr for attr in dir(dataset) if not attr.startswith('_')]
        for attr in attrs:
            print(f"  - {attr}")
        print()

        # Try different methods to access qrels
        qrels_by_query = {}
        non_zero_count = 0

        qrels_method = None
        for method_name in ['qrels_iter', 'qrels', 'scoreddocs_iter', 'docpairs_iter']:
            if hasattr(dataset, method_name):
                qrels_method = method_name
                print(f"Found qrels method: {method_name}")
                break

        # Also check if there's an official qrels dataset variant
        print("\nChecking for official qrels dataset...")
        qrels_variants = [
            'msmarco-passage-v2/trec-dl-2023/judged',
            'msmarco-passage-v2/trec-dl-2023-qrels',
            'msmarco-passage/trec-dl-2023/judged',
        ]

        official_qrels_dataset = None
        for variant in qrels_variants:
            try:
                print(f"  Trying {variant}...")
                test_ds = ir_datasets.load(variant)
                official_qrels_dataset = test_ds
                print(f"  ✓ Found official qrels at {variant}")
                break
            except:
                pass

        if official_qrels_dataset and hasattr(official_qrels_dataset, 'qrels_iter'):
            print("\n" + "=" * 80)
            print("OFFICIAL HUMAN RELEVANCE JUDGMENTS (QRELS)")
            print("=" * 80)

            official_qrels = {}
            for qrel in official_qrels_dataset.qrels_iter():
                if qrel.query_id not in official_qrels:
                    official_qrels[qrel.query_id] = []
                official_qrels[qrel.query_id].append({
                    'doc_id': qrel.doc_id,
                    'relevance': qrel.relevance
                })

            print(f"Official qrels available for {len(official_qrels)} queries")
            matching_official = set(official_qrels.keys()) & matching_queries
            print(f"Matching queries with official qrels: {len(matching_official)}")

            if matching_official:
                sample_qid = list(matching_official)[0]
                print(f"\nSample official qrels for query {sample_qid}:")
                for qrel in official_qrels[sample_qid][:5]:
                    print(f"  Doc: {qrel['doc_id']}, Grade: {qrel['relevance']}")

        if qrels_method is None:
            print("⚠ No qrels/relevance judgments found in this dataset")
            print("This might mean:")
            print("  1. TREC DL 2023 qrels not yet released in ir_datasets")
            print("  2. Need to download them separately from TREC website")
            print("  3. Dataset uses different structure")
        else:
            print(f"Loading TREC DL 2023 qrels using {qrels_method}...")

            qrels_iter = getattr(dataset, qrels_method)
            if callable(qrels_iter):
                qrels_iter = qrels_iter()

            for qrel in qrels_iter:
                qid = getattr(qrel, 'query_id', None)
                doc_id = getattr(qrel, 'doc_id', None)
                relevance = getattr(qrel, 'relevance', getattr(qrel, 'score', 0))

                if qid is None:
                    continue

                if qid not in qrels_by_query:
                    qrels_by_query[qid] = []
                qrels_by_query[qid].append({
                    'doc_id': doc_id,
                    'relevance': relevance
                })
                if relevance > 0:
                    non_zero_count += 1

        print(f"TREC DL 2023 has qrels for {len(qrels_by_query)} queries")
        print(f"Non-zero relevance judgments: {non_zero_count}")

        # Check if our matching queries have qrels
        matching_with_qrels = set(qrels_by_query.keys()) & matching_queries
        print(f"Matching queries with qrels: {len(matching_with_qrels)}")

        if matching_with_qrels:
            print("\nSample qrels for matching queries:")
            sample_qid = list(matching_with_qrels)[0]
            print(f"\nQuery ID: {sample_qid}")
            for qrel in qrels_by_query[sample_qid][:5]:
                print(f"  Doc: {qrel['doc_id']}, Relevance: {qrel['relevance']}")

        print()
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)

        match_percentage = (len(matching_queries) / len(query_ids) * 100) if query_ids else 0
        print(f"Query match rate: {match_percentage:.1f}%")

        if match_percentage > 90:
            print("✓ Strong evidence this data comes from TREC DL 2023")
            if non_zero_count > 0:
                print(f"✓ Found {non_zero_count} scoreddocs (retrieval scores) in ir_datasets")
                print("\n⚠ IMPORTANT:")
                print("  - scoreddocs are RETRIEVAL SCORES from baseline ranker")
                print("  - These are NOT the same as human relevance judgments (qrels)")
                print("  - For evaluation, you need official TREC qrels with grades 0-3")
                print("\nNext steps:")
                print("  1. Check if official qrels appeared above")
                print("  2. If not, visit: https://trec.nist.gov/data/deep.html")
                print("  3. Look for TREC 2023 DL Track qrels download")
            else:
                print("✗ No non-zero relevance labels found")
        elif match_percentage > 50:
            print("⚠ Partial match - data might be from TREC DL 2023 subset")
        else:
            print("✗ Low match rate - data likely from different source")

    except ImportError:
        print("ir_datasets not installed. Install with: pip install ir-datasets")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
