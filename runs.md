# INGEST
# INFER
infer \
  --input ./artifacts/runs/ingest/test/20251111_075759_llmjudge-json/normalized_dataset.json \
  --io-cfg json \
  --model-cfg gpt-oss-20b-free \
  --prompt-cfg thomas-simple \
  --limit 3

# AGGREGATE
# EVALUATE