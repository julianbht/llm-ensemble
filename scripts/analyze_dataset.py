#!/usr/bin/env python3
"""Quick dataset statistics for LLM Judge Challenge 2024 dataset."""

import json
from pathlib import Path
from collections import Counter


def main():
    base_dir = Path("/home/jn/dev/llm-ensemble/datasets/llm_judge_challenge_experiment")
    qrels_path = base_dir / "llm4eval_test_qrel_2024_recovered.txt"

    # Track unique queries and documents referenced in qrels
    unique_queries = set()
    unique_docs = set()
    qrel_count = 0
    label_counter = Counter()

    with qrels_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Format: qid iteration docid relevance
            parts = line.split()
            if len(parts) != 4:
                continue

            qid = parts[0]
            docid = parts[2]
            relevance = int(parts[3])

            unique_queries.add(qid)
            unique_docs.add(docid)
            qrel_count += 1
            label_counter[relevance] += 1

    # Print results
    print("=" * 60)
    print("LLM Judge Challenge 2024 Dataset Statistics")
    print("(Test Qrel Split)")
    print("=" * 60)
    print(f"Data directory: {base_dir}")
    print()
    print(f"Total queries:    {len(unique_queries):,}")
    print(f"Total documents:  {len(unique_docs):,}")
    print(f"Total qrels:      {qrel_count:,}")
    print()
    print("Label distribution:")
    for label in sorted(label_counter.keys()):
        count = label_counter[label]
        percentage = (count / qrel_count * 100) if qrel_count > 0 else 0
        print(f"  Label {label}: {count:,} ({percentage:.2f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
