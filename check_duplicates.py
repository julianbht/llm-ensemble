#!/usr/bin/env python3
"""Check for duplicate (query, document) pairs with different gold scores in qrels."""

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

def check_content_duplicates(data_dir: Path):
    """Check for qrels entries with different external IDs but same content."""

    queries_path = data_dir / "llm4eval_query_2024.txt"
    docs_path = data_dir / "llm4eval_document_2024.jsonl"
    qrels_path = data_dir / "llm4eval_dev_qrel_2024.txt"

    # Load queries and documents
    print("Loading queries and documents...")
    queries = read_queries(queries_path)
    docs = read_documents(docs_path)

    print(f"Loaded {len(queries)} queries and {len(docs)} documents")
    print()

    # Process qrels and track content hashes
    # Map (query_hash, doc_hash) -> list of (external_qid, external_docid, relevance, line_num)
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

            # Get text content
            query_text = queries.get(qid)
            doc_text = docs.get(docid)

            if query_text is None or doc_text is None:
                continue

            # Compute content hashes
            query_hash = compute_content_hash(query_text)
            doc_hash = compute_content_hash(doc_text)

            content_pairs[(query_hash, doc_hash)].append((qid, docid, int(rel), line_num))

    # Find content duplicates
    unique_content_pairs = len(content_pairs)
    content_duplicates = []

    for (q_hash, d_hash), entries in content_pairs.items():
        if len(entries) > 1:
            content_duplicates.append((q_hash, d_hash, entries))

    # Print summary
    print(f"=== Content-Based Deduplication Analysis ===")
    print(f"Total qrels entries: {total_lines}")
    print(f"Unique content (query, doc) pairs: {unique_content_pairs}")
    print(f"Content duplicates (different external IDs, same content): {len(content_duplicates)}")
    print(f"Total duplicate entries: {sum(len(e[2]) - 1 for e in content_duplicates)}")
    print()

    if content_duplicates:
        print("=== CONTENT DUPLICATES (first 5) ===")
        for i, (q_hash, d_hash, entries) in enumerate(content_duplicates[:5]):
            print(f"\n{i+1}. Content pair appears {len(entries)} times:")
            print(f"   Query hash: {q_hash[:16]}...")
            print(f"   Doc hash: {d_hash[:16]}...")

            # Show first entry's full text
            first_qid, first_docid, first_rel, first_line = entries[0]
            query_text = queries[first_qid]
            doc_text = docs[first_docid]
            print(f"   Query text: {query_text[:100]}..." if len(query_text) > 100 else f"   Query text: {query_text}")
            print(f"   Doc text: {doc_text[:100]}..." if len(doc_text) > 100 else f"   Doc text: {doc_text}")
            print()
            print("   External IDs:")
            for qid, docid, rel, line_num in entries:
                print(f"     Line {line_num}: query={qid}, doc={docid}, score={rel}")

if __name__ == "__main__":
    data_dir = Path("data")

    check_content_duplicates(data_dir)
