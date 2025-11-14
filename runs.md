# INGEST
ingest \
> --input ./data \
> --io-cfg llm_judge_challenge_json \
# INFER
infer \
  --input ./artifacts/runs/ingest/test/20251111_075759_llmjudge-json/judging_samples.json \
  --io-cfg sql \
  --model-cfg gpt-oss-20b-free \
  --prompt-cfg thomas-simple \
  --limit 3

# AGGREGATE
# EVALUATE