# TREC Qrels Recovery Pipeline

This pipeline recovers the withheld test set labels from the LLM4Eval challenge by matching queries with TREC 2023 Deep Learning Track data.

## Quick Start

```bash
# Run all steps
./run_all_verification_steps.sh
```

## Step-by-Step

```bash
# 1. Verify data source (requires ir-datasets)
python3 step1_verify_data_source.py

# 2. Download official TREC qrels
python3 step2_download_trec_qrels.py

# 3. Extract challenge qrels
python3 step3_extract_challenge_qrels.py

# 4. Compute agreement metrics
python3 step4_compute_agreement_metrics.py
```

## Dependencies

```bash
pip install -r requirements_verification.txt
```

## Output

- Reads from: `../../data/llm_judge_challenge/`
- Writes to: `../../data/llm_judge_challenge_qrels_recovered/`

## Results

- **Step 1**: 100% query match with TREC 2023 DL
- **Step 2**: Downloads 22,327 qrels
- **Step 3**: Extracts 13,690 qrels for 50 queries
- **Step 4**: Cohen's κ ≈ 0.19, Krippendorff's α ≈ 0.38

## Documentation

See `../../docs/trec_qrels_verification.md` for full methodology.
