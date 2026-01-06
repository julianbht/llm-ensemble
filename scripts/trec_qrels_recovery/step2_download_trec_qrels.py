#!/usr/bin/env python3
"""
Step 2: Download official TREC 2023 DL Track passage qrels from NIST.

Downloads the official human relevance judgments for all 700 queries in TREC 2023.
"""

from pathlib import Path
import urllib.request


TREC_QRELS_URL = "https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt"


def main():
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'llm_judge_challenge_test'
    output_file = data_dir / 'trec_2023_passage_qrels_official.txt'

    print("=" * 80)
    print("STEP 2: DOWNLOAD TREC 2023 QRELS")
    print("=" * 80)
    print()

    print(f"Source: {TREC_QRELS_URL}")
    print(f"Output: {output_file}")
    print()

    if output_file.exists():
        print("⚠ File already exists. Skipping download.")
        print("  Delete the file first if you want to re-download.")
    else:
        print("Downloading...")
        try:
            urllib.request.urlretrieve(TREC_QRELS_URL, output_file)
            print("✓ Download complete")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return 1

    # Verify file
    print()
    print("Verifying file...")
    line_count = 0
    sample_lines = []

    with open(output_file) as f:
        for i, line in enumerate(f):
            line_count += 1
            if i < 5:
                sample_lines.append(line.strip())

    print(f"  Total lines: {line_count:,}")
    print()
    print("  Sample qrels:")
    for line in sample_lines:
        parts = line.split()
        if len(parts) == 4:
            qid, _, doc_id, grade = parts
            print(f"    Query: {qid}, Doc: {doc_id[:30]}..., Grade: {grade}")

    print()
    print("✓ TREC 2023 qrels downloaded successfully")

    return 0


if __name__ == '__main__':
    exit(main())
