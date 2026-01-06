#!/bin/bash
# Master script to run all verification steps in sequence

set -e  # Exit on error

echo "=========================================="
echo "TREC 2023 Qrels Verification Pipeline"
echo "=========================================="
echo ""

# Check if virtual environment is activated
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠ Warning: Virtual environment not activated"
    echo "  Activate with: source .venv/bin/activate"
    echo ""
fi

# Step 1: Verify data source
echo "Running Step 1: Verify data source..."
python3 scripts/step1_verify_data_source.py
if [ $? -ne 0 ]; then
    echo "✗ Step 1 failed"
    exit 1
fi
echo ""

# Step 2: Download TREC qrels
echo "Running Step 2: Download TREC qrels..."
python3 scripts/step2_download_trec_qrels.py
if [ $? -ne 0 ]; then
    echo "✗ Step 2 failed"
    exit 1
fi
echo ""

# Step 3: Extract challenge qrels
echo "Running Step 3: Extract challenge qrels..."
python3 scripts/step3_extract_challenge_qrels.py
if [ $? -ne 0 ]; then
    echo "✗ Step 3 failed"
    exit 1
fi
echo ""

# Step 4: Compute agreement metrics
echo "Running Step 4: Compute agreement metrics..."
python3 scripts/step4_compute_agreement_metrics.py
if [ $? -ne 0 ]; then
    echo "✗ Step 4 failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ All verification steps completed"
echo "=========================================="
