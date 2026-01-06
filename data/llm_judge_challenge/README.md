# LLM Judge Challenge - Original Data

This directory contains the **original, unmodified files** from the LLM4Eval challenge.

**Do not modify these files.** They serve as the source of truth for the challenge data.

## Files

- `docid_to_docidx.txt` - Document ID mapping (11,621 documents)
- `qid_to_qidx.txt` - Query ID mapping (50 queries)
- `llm4eval_query_2024.txt` - Query texts
- `llm4eval_dev_qrel_2024.txt` - Dev set qrels (with labels)
- `llm4eval_test_qrel_2024.txt` - Test set qrels (withheld - all zeros)
- `llm4eval_document_2024.jsonl` - Document corpus
- `NISTRetrieval-instruct0.txt` - Example submission

## Usage

For work involving the recovered ground truth labels, use `../llm_judge_challenge_qrels_recovered/` instead.
