# LLM Judge Challenge - Test Set with Recovered Qrels

This directory contains all original challenge files **plus recovered ground truth labels** from TREC 2023 Deep Learning Track.

## Key Files

### Recovered Ground Truth
- **llm4eval_official_qrels_2023.txt** - **Use this for evaluation**
  - 13,690 query-document-grade judgments
  - Covers all 50 test queries
  - Grades 0-3 (official TREC human judgments)

### Full TREC Dataset
- **trec_2023_passage_qrels_official.txt**
  - Complete TREC 2023 qrels (22,327 judgments for 700 queries)
  - Superset of the challenge queries

### Original Challenge Files
All files from `../llm_judge_challenge/` are also present here.

## How to Use

```python
# Load recovered qrels for evaluation
qrels = {}
with open('data/llm_judge_challenge_test/llm4eval_official_qrels_2023.txt') as f:
    for line in f:
        qid, _, doc_id, grade = line.strip().split()
        qrels[(qid, doc_id)] = int(grade)
```

## Regenerating

To regenerate the recovered qrels:
```bash
./scripts/trec_qrels_recovery/run_all_verification_steps.sh
```

See `../../docs/trec_qrels_verification.md` for methodology.
