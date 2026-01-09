# LLM Judge Challenge - Recovered Qrels

This directory contains **ONLY generated files** from the TREC qrels recovery pipeline.

For original challenge files, see `../llm_judge_challenge/`.

## Key Files

### **USE THIS:** Challenge Format (Anonymized Indices)
- **llm4eval_test_qrel_2024_recovered.txt** - **Main recovered qrels file**
  - Format: `query_idx iteration doc_idx grade` (e.g., `q0 0 p123 2`)
  - 4,423 judgments with actual grades (0-3)
  - Exact same query-doc pairs as original withheld test set
  - Use with `llm4eval_query_2024.txt` and `llm4eval_document_2024.jsonl`
  - **This is what you want for fair comparison with LLM Judge Challenge!**

- **llm4eval_test_qrel_2024_recovered_superset.txt** - **Optional extended test set**
  - Format: `query_idx iteration doc_idx grade`
  - 11,718 judgments for 50 queries
  - Contains all original pairs plus additional queries from TREC 2023
  - Use if you want more test data

### TREC Format (Real IDs)
- **trec_2023_challenge_subset.txt**
  - Format: `query_id iteration doc_id grade` (e.g., `2001459 0 msmarco_passage_00_168095376 2`)
  - 13,690 judgments with real TREC IDs
  - Used for validation against paper

- **trec_2023_passage_qrels_official.txt**
  - Complete TREC 2023 qrels (22,327 judgments for 700 queries)
  - Superset containing all challenge queries plus additional queries


## How to Use

### Recommended: Use Main Recovered File
Replace the withheld test qrels file:
```python
# Load challenge format qrels (anonymized indices)
qrels = {}
with open('datasets/llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered.txt') as f:
    for line in f:
        query_idx, iteration, doc_idx, grade = line.strip().split()
        qrels[(query_idx, doc_idx)] = int(grade)
# Now use with llm4eval_query_2024.txt and llm4eval_document_2024.jsonl
```

### Optional: Use Extended Superset
If you want more test data (50 queries instead of 25):
```python
qrels = {}
with open('datasets/llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered_superset.txt') as f:
    for line in f:
        query_idx, iteration, doc_idx, grade = line.strip().split()
        qrels[(query_idx, doc_idx)] = int(grade)
```

### Alternative: TREC Format (Real IDs)
If you need real TREC IDs instead of anonymized indices:
```python
qrels = {}
with open('datasets/llm_judge_challenge_qrels_recovered/trec_2023_challenge_subset.txt') as f:
    for line in f:
        qid, _, doc_id, grade = line.strip().split()
        qrels[(qid, doc_id)] = int(grade)
```

## Using With Your System

If your ingest expects all files in one directory:

**Option A: Copy the recovered qrels**
```bash
cp datasets/llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered.txt \
   datasets/llm_judge_challenge/
```

**Option B: Symlink**
```bash
ln -s ../llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered.txt \
      datasets/llm_judge_challenge/
```

Then point your ingest to `datasets/llm_judge_challenge/`.

## Verification

These qrels have been validated against the LLM Judge Challenge paper (page 6):
- Cohen's κ = 0.1877 ✓
- Krippendorff's α = 0.3819 ✓

The agreement metrics match, confirming these are the correct gold labels.

## Regenerating

To regenerate and re-verify the recovered qrels:
```bash
./scripts/trec_qrels_recovery/run_all_verification_steps.sh
```

See `../../docs/trec_qrels_verification.md` for methodology.
