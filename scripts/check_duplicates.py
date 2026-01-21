#!/usr/bin/env python3
"""Check for duplicate (query, document) pairs in qrels.

Reports three types of duplicates:
1. Exact qrel duplicates: Same (qid, docid) pair appearing multiple times
2. Content duplicates: Different external IDs but identical text content
3. Conflicting labels: Same content but different relevance scores
"""

from pathlib import Path
from collections import defaultdict
import json
import hashlib


def read_queries(queries_path: Path) -> dict[str, str]:
    """Read queries file and return dict of external_id -> query_text."""
    queries = {}
    with queries_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                qid, qtext = parts
                queries[qid.strip()] = qtext.strip()
    return queries


def read_documents(docs_path: Path) -> dict[str, str]:
    """Read documents file and return dict of external_id -> doc_text."""
    docs = {}
    with docs_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs[obj["docid"]] = obj["doc"]
    return docs


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of text content."""
    return hashlib.sha256(text.encode()).hexdigest()


def check_duplicates(data_dir: Path):
    """Check for all types of duplicates in qrels."""

    queries_path = data_dir / "llm4eval_query_2024.txt"
    docs_path = data_dir / "llm4eval_document_2024.jsonl"

    # Prefer recovered test qrels (same logic as ingest reader)
    qrels_path = data_dir / "llm4eval_test_qrel_2024_recovered.txt"
    if not qrels_path.exists():
        qrels_path = data_dir / "llm4eval_dev_qrel_2024.txt"
    if not qrels_path.exists():
        qrels_path = data_dir / "llm4eval_test_qrel_2024.txt"

    # Print paths being used
    print(f"Dataset directory: {data_dir.resolve()}")
    print(f"  Queries: {queries_path}")
    print(f"  Documents: {docs_path}")
    print(f"  Qrels: {qrels_path}")
    print()

    # Load queries and documents
    print("Loading queries and documents...")
    queries = read_queries(queries_path)
    docs = read_documents(docs_path)
    print(f"Loaded {len(queries)} queries and {len(docs)} documents")
    print()

    # Track duplicates
    # 1. Exact qrel duplicates: (qid, docid) -> list of (relevance, line_num)
    exact_qrel_pairs = defaultdict(list)
    # 2. Content duplicates: (query_hash, doc_hash) -> list of (qid, docid, relevance, line_num)
    content_pairs = defaultdict(list)

    total_lines = 0
    with qrels_path.open("r") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total_lines += 1
            parts = line.split()

            if len(parts) == 3:
                qid, rel, docid = parts
            elif len(parts) == 4:
                qid, _, docid, rel = parts
            else:
                continue

            rel = int(rel)

            # Track exact qrel duplicates
            exact_qrel_pairs[(qid, docid)].append((rel, line_num))

            # Get text content for content-based deduplication
            query_text = queries.get(qid)
            doc_text = docs.get(docid)

            if query_text is None or doc_text is None:
                continue

            # Compute content hashes
            query_hash = compute_content_hash(query_text)
            doc_hash = compute_content_hash(doc_text)
            content_pairs[(query_hash, doc_hash)].append((qid, docid, rel, line_num))

    # === Analysis ===
    print("=" * 60)
    print("DUPLICATE ANALYSIS")
    print("=" * 60)
    print(f"Total qrels entries: {total_lines}")
    print()

    # 1. Exact qrel duplicates
    exact_duplicates = [(k, v) for k, v in exact_qrel_pairs.items() if len(v) > 1]
    exact_dup_entries = sum(len(v) - 1 for _, v in exact_duplicates)
    print(f"1. EXACT QREL DUPLICATES (same qid+docid)")
    print(f"   Unique (qid, docid) pairs: {len(exact_qrel_pairs)}")
    print(f"   Pairs appearing multiple times: {len(exact_duplicates)}")
    print(f"   Total duplicate entries: {exact_dup_entries}")

    if exact_duplicates:
        print("\n   Examples (first 3):")
        for i, ((qid, docid), entries) in enumerate(exact_duplicates[:3]):
            scores = [e[0] for e in entries]
            lines = [e[1] for e in entries]
            conflict = "CONFLICTING SCORES!" if len(set(scores)) > 1 else "same score"
            print(f"     {i+1}. qid={qid}, docid={docid} appears {len(entries)} times")
            print(f"        Scores: {scores} ({conflict})")
            print(f"        Lines: {lines}")
    print()

    # 2. Content duplicates
    content_duplicates = [(k, v) for k, v in content_pairs.items() if len(v) > 1]
    content_dup_entries = sum(len(v) - 1 for _, v in content_duplicates)
    print(f"2. CONTENT DUPLICATES (different IDs, same text)")
    print(f"   Unique content (query, doc) pairs: {len(content_pairs)}")
    print(f"   Content pairs with multiple IDs: {len(content_duplicates)}")
    print(f"   Total duplicate entries: {content_dup_entries}")

    if content_duplicates:
        print("\n   Examples (first 3):")
        for i, ((q_hash, d_hash), entries) in enumerate(content_duplicates[:3]):
            first_qid, first_docid, _, _ = entries[0]
            query_text = queries[first_qid][:60] + "..." if len(queries[first_qid]) > 60 else queries[first_qid]
            print(f"     {i+1}. Query: \"{query_text}\"")
            print(f"        Appears with {len(entries)} different ID combinations:")
            for qid, docid, rel, line_num in entries[:3]:
                print(f"          Line {line_num}: qid={qid}, docid={docid}, score={rel}")
            if len(entries) > 3:
                print(f"          ... and {len(entries) - 3} more")
    print()

    # 3. Conflicting labels (content duplicates with different scores)
    conflicting = []
    for (q_hash, d_hash), entries in content_pairs.items():
        scores = set(e[2] for e in entries)
        if len(scores) > 1:
            conflicting.append((q_hash, d_hash, entries, scores))

    print(f"3. CONFLICTING LABELS (same content, different scores)")
    print(f"   Content pairs with conflicting scores: {len(conflicting)}")

    if conflicting:
        print("\n   Examples (first 5):")
        for i, (q_hash, d_hash, entries, scores) in enumerate(conflicting[:5]):
            first_qid, first_docid, _, _ = entries[0]
            query_text = queries[first_qid][:60] + "..." if len(queries[first_qid]) > 60 else queries[first_qid]
            print(f"     {i+1}. Query: \"{query_text}\"")
            print(f"        Conflicting scores: {sorted(scores)}")
            for qid, docid, rel, line_num in entries:
                print(f"          Line {line_num}: qid={qid}, docid={docid}, score={rel}")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total qrels entries: {total_lines}")
    print(f"Exact duplicates (same IDs): {exact_dup_entries}")
    print(f"Content duplicates (different IDs, same text): {content_dup_entries}")
    print(f"Conflicting labels (same content, different scores): {len(conflicting)}")
    print(f"Clean entries after deduplication: {len(content_pairs)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1])
    else:
        # Default to the dataset used by the infer pipeline
        data_dir = Path("datasets/llm_judge_challenge_experiment")

    check_duplicates(data_dir)
