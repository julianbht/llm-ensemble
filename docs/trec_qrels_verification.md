# TREC 2023 DL Qrels Verification for LLM4Eval Challenge

## Objective

The LLM4Eval challenge withheld test set gold labels (all zeros in `llm4eval_test_qrel_2024.txt`).
We hypothesized that the test data comes from **TREC 2023 Deep Learning Track** and attempted
to recover the gold labels from the official TREC qrels.

## Methodology

### Step 1: Verify Data Source

**Goal**: Confirm that the challenge test queries come from TREC 2023 DL Track

**Input**:
- `datasets/llm_judge_challenge/qid_to_qidx.txt` - Mapping from actual query IDs to anonymized indices (q0, q1, ...)
- `datasets/llm_judge_challenge/docid_to_docidx.txt` - Mapping from actual doc IDs to anonymized indices (p0, p1, ...)

**Process**:
- Loaded 50 query IDs from the mapping file
- Matched against TREC 2023 DL passage queries using `ir_datasets` library
- Query IDs format: `2040064`, `3100289`, etc. (numeric IDs from MS MARCO v2)

**Result**:
- **50/50 queries matched (100%)** → Strong evidence data comes from TREC 2023 DL Track

**Script**: `scripts/trec_qrels_recovery/step1_verify_data_source.py`

---

### Step 2: Download Official TREC Qrels

**Goal**: Obtain the official human relevance judgments from TREC 2023

**Source**: NIST official TREC Deep Learning Track 2023
- URL: https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt
- Dataset: TREC 2023 DL Track Passage Ranking

**Format**: Standard TREC qrels format
```
<query_id> <iteration> <doc_id> <relevance_grade>
```

**Relevance grades**: 0-3 scale
- 0: Not relevant
- 1: Relevant
- 2: Highly relevant
- 3: Perfectly relevant

**Output**: `datasets/llm_judge_challenge_qrels_recovered/trec_2023_passage_qrels_official.txt`
- Total: 22,327 judgments across 700 queries (full TREC 2023 set)

**Script**: `scripts/trec_qrels_recovery/step2_download_trec_qrels.py`

---

### Step 3: Extract Challenge-Specific Qrels

**Goal**: Filter TREC qrels to only include the 50 challenge queries

**Input**:
- `datasets/llm_judge_challenge_qrels_recovered/trec_2023_passage_qrels_official.txt` - Full TREC 2023 qrels
- `datasets/llm_judge_challenge/qid_to_qidx.txt` - Challenge query mapping

**Process**:
- Extracted all qrels where query_id matches one of the 50 challenge queries
- Kept original query IDs and doc IDs (not anonymized indices)

**Result**:
- **13,690 query-document-grade triplets** for 50 queries
- All 50 challenge queries have gold labels
- Grade distribution: 62.4% grade 0, 19.3% grade 1, 10.5% grade 2, 7.8% grade 3

**Output**: `datasets/llm_judge_challenge_qrels_recovered/trec_2023_challenge_subset.txt`

**Script**: `scripts/trec_qrels_recovery/step3_extract_challenge_qrels.py`

---

### Step 4: Compute Agreement Metrics

**Goal**: Validate our qrels by computing agreement metrics with a published submission

**Input**:
- `datasets/llm_judge_challenge/NISTRetrieval-instruct0.txt` - LLM judge submission (anonymized format: q0, p0, etc.)
- `datasets/llm_judge_challenge_qrels_recovered/trec_2023_challenge_subset.txt` - Our extracted gold labels
- `datasets/llm_judge_challenge/qid_to_qidx.txt`, `datasets/llm_judge_challenge/docid_to_docidx.txt` - ID mappings

**Process**:
1. Load submission predictions (4,423 query-document-grade triplets)
2. Map anonymized indices (q0, p0) to actual IDs using mapping files
3. Align predictions with gold labels on (query_id, doc_id) pairs
4. Compute instance-level agreement metrics:
   - **Cohen's Kappa (κ)** - Agreement corrected for chance
   - **Krippendorff's Alpha (α)** - Generalized agreement for ordinal data

**Results**:
```
Cohen's κ (Kappa):              0.1877
Krippendorff's α (Alpha):       0.3819
Exact agreement:                42.8% (1895/4423)
```

**Validation**:
These metrics match the published values from the LLM Judge Challenge paper (page 6), confirming we have the correct gold labels.

**Interpretation**:
- Metrics show **fair to moderate agreement** between LLM judge and human gold labels
- This is reasonable for LLM-as-judge tasks (not expected to be perfect)
- These metrics are computed at the **instance level** (query-document pairs)
- **Matching these values validates** that our recovered qrels are correct

**Script**: `scripts/trec_qrels_recovery/step4_compute_agreement_metrics.py`

---

### Step 5: Convert to Challenge Format

**Goal**: Convert TREC format qrels (real IDs) to challenge format (anonymized indices)

**Input**:
- `datasets/llm_judge_challenge_qrels_recovered/trec_2023_challenge_subset.txt` - Qrels with real IDs
- `datasets/llm_judge_challenge/qid_to_qidx.txt`, `datasets/llm_judge_challenge/docid_to_docidx.txt` - ID mappings

**Process**:
1. Load TREC format qrels with real query and document IDs
2. Map each ID to its anonymized index (qX, pX)
3. Output in challenge format: `query_idx iteration doc_idx grade`

**Result**:
- **13,690 judgments** in challenge format
- Compatible with `llm4eval_query_2024.txt` and `llm4eval_document_2024.jsonl`
- Drop-in replacement for `llm4eval_test_qrel_2024.txt`

**Output**: `datasets/llm_judge_challenge_qrels_recovered/llm4eval_test_qrel_2024_recovered.txt`

**Script**: `scripts/trec_qrels_recovery/step5_convert_to_challenge_format.py`

---

## Key Assumption

**We assume** that the TREC 2023 gold labels we extracted are the same labels used by the
LLM4Eval challenge organizers. This is supported by:

1. **Perfect query ID match** (50/50)
2. **Document IDs use TREC format** (`msmarco_passage_XX_YYYYYY`)
3. **Challenge explicitly mentions TREC 2023** in documentation/papers

However, we **cannot be 100% certain** without:
- Official confirmation from challenge organizers
- Exact metric replication (requires system-level evaluation with multiple runs)

---

## Limitations

1. **System ranking metrics not computed**: The paper reports Kendall's τ=0.944 and
   Spearman's ρ=0.9907, which are **system-level** ranking correlations requiring
   multiple system runs. We only computed **instance-level** agreement metrics.

2. **Single submission verified**: Only validated with one submission (NISTRetrieval-instruct0).
   More submissions would strengthen confidence.

3. **No official ground truth release**: The challenge has not officially released test labels,
   so this remains a hypothesis until confirmed.

---

## Files Generated

```
data/
├── llm_judge_challenge/                    # Original challenge files ONLY
│   ├── docid_to_docidx.txt
│   ├── qid_to_qidx.txt
│   ├── llm4eval_*_2024.txt
│   └── NISTRetrieval-instruct0.txt
└── llm_judge_challenge_qrels_recovered/    # Generated files ONLY
    ├── trec_2023_passage_qrels_official.txt         # Full TREC 2023 (22,327, real IDs)
    ├── trec_2023_challenge_subset.txt               # Challenge subset (13,690, real IDs)
    └── llm4eval_test_qrel_2024_recovered.txt        # Challenge format (13,690, q/p indices) ⭐

scripts/trec_qrels_recovery/
├── step1_verify_data_source.py             # Verify queries match TREC 2023
├── step2_download_trec_qrels.py            # Download official qrels
├── step3_extract_challenge_qrels.py        # Filter to challenge queries
├── step4_compute_agreement_metrics.py      # Compute κ and α (validation)
├── step5_convert_to_challenge_format.py    # Convert to anonymized format
└── run_all_verification_steps.sh           # Master script
```

⭐ **Use this file** for evaluation - it's the drop-in replacement for the withheld test qrels.

---

## Reproducibility

To reproduce this analysis:

```bash
# Install dependencies
pip install -r scripts/trec_qrels_recovery/requirements_verification.txt

# Run all steps
./scripts/trec_qrels_recovery/run_all_verification_steps.sh

# Or run individually
python3 scripts/trec_qrels_recovery/step1_verify_data_source.py
python3 scripts/trec_qrels_recovery/step2_download_trec_qrels.py
python3 scripts/trec_qrels_recovery/step3_extract_challenge_qrels.py
python3 scripts/trec_qrels_recovery/step4_compute_agreement_metrics.py
python3 scripts/trec_qrels_recovery/step5_convert_to_challenge_format.py
```

---

## References

**LLM Judge Challenge Paper**:
- Source: `scripts/trec_qrels_recovery/llm-judge-challenge.pdf` (page 6)
- Contains the validation metrics (Cohen's κ = 0.1877, Krippendorff's α = 0.3819)

**Citations**:
- **TREC 2023 Deep Learning Track**: Craswell et al., TREC 2023
- **MS MARCO v2 Dataset**: Bajaj et al.
- **LLM4Eval Challenge**: [Challenge organizers]
