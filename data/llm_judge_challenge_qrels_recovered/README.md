# LLM Judge Challenge - Recovered Qrels

This directory contains recovered ground truth labels from TREC 2023 Deep Learning Track in multiple formats.

## Key Files

### **USE THIS:** Challenge Format (Anonymized Indices)
- **llm4eval_test_qrel_2024_recovered.txt** - **Drop-in replacement for test qrels**
  - Format: `query_idx iteration doc_idx grade` (e.g., `q0 0 p123 2`)
  - 13,690 judgments with actual grades (0-3)
  - Use with `llm4eval_query_2024.txt` and `llm4eval_document_2024.jsonl`
  - **This is what you want for evaluation!**

### TREC Format (Real IDs)
- **trec_2023_challenge_subset.txt**
  - Format: `query_id iteration doc_id grade` (e.g., `2001459 0 msmarco_passage_00_168095376 2`)
  - Same 13,690 judgments but with real TREC IDs
  - Used for validation against paper

- **trec_2023_passage_qrels_official.txt**
  - Complete TREC 2023 qrels (22,327 judgments for 700 queries)
  - Superset containing all challenge queries plus additional queries

### Original Challenge Files
Also contains copies of files from `../llm_judge_challenge/` for convenience.

## How to Use

### Option 1: Direct Replacement (Recommended)
Replace the withheld test qrels file:
```python
# Load challenge format qrels (anonymized indices)
qrels = {}
with open('data/llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered.txt') as f:
    for line in f:
        query_idx, iteration, doc_idx, grade = line.strip().split()
        qrels[(query_idx, doc_idx)] = int(grade)
# Now use with llm4eval_query_2024.txt and llm4eval_document_2024.jsonl
```

### Option 2: TREC Format (Real IDs)
If you need real TREC IDs instead of anonymized indices:
```python
qrels = {}
with open('data/llm_judge_challenge_qrels_recovered/trec_2023_challenge_subset.txt') as f:
    for line in f:
        qid, _, doc_id, grade = line.strip().split()
        qrels[(qid, doc_id)] = int(grade)
```

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
