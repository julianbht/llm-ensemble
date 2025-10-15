#!/usr/bin/env python3
"""Example log analysis script demonstrating structured JSON log queries.

This script shows how to analyze logs from LLM Ensemble CLI runs.
You can easily filter, aggregate, and visualize operational data.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def load_logs(log_file: Path) -> list[dict]:
    """Load JSONL logs into a list of dicts."""
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def analyze_ingest_run(log_file: Path):
    """Analyze logs from an ingest run."""
    logs = load_logs(log_file)

    print(f"=== Ingest Run Analysis: {log_file.parent.name} ===\n")

    # Summary stats
    events = [log["event"] for log in logs]
    print(f"Total log entries: {len(logs)}")
    print(f"Events: {', '.join(events)}\n")

    # Find start/end times
    start = next((log for log in logs if log["event"] == "ingest_started"), None)
    end = next((log for log in logs if log["event"] == "ingest_completed"), None)

    if start and end:
        print(f"Dataset: {start.get('dataset')}")
        print(f"Data dir: {start.get('data_dir')}")
        print(f"Limit: {start.get('limit', 'None')}")
        print(f"Samples processed: {end.get('sample_count')}")
        print(f"Git SHA: {start.get('git_sha')}")
        print(f"Started: {start['timestamp']}")
        print(f"Completed: {end['timestamp']}\n")


def analyze_infer_run(log_file: Path):
    """Analyze logs from an infer run."""
    logs = load_logs(log_file)

    print(f"=== Infer Run Analysis: {log_file.parent.name} ===\n")

    # Summary stats
    successes = [log for log in logs if log.get("event") == "inference_success"]
    failures = [log for log in logs if log.get("event") == "inference_failed"]

    print(f"Total log entries: {len(logs)}")
    print(f"Inference successes: {len(successes)}")
    print(f"Inference failures: {len(failures)}\n")

    # Find start/end
    start = next((log for log in logs if log["event"] == "inference_started"), None)
    end = next((log for log in logs if log["event"] == "inference_completed"), None)

    if start:
        print(f"Model: {start.get('model')}")
        print(f"Provider: {start.get('provider')}")
        print(f"Samples: {start.get('num_samples')}")
        print(f"Prompt: {start.get('prompt_template')}")
        print(f"Git SHA: {start.get('git_sha')}\n")

    if end:
        print(f"Judgements: {end.get('judgement_count')}")
        print(f"Errors: {end.get('error_count')}")
        print(f"Avg latency: {end.get('avg_latency_ms'):.2f}ms\n")

    # Latency stats (if available)
    if successes:
        latencies = [log["latency_ms"] for log in successes if "latency_ms" in log]
        if latencies:
            print(f"Latency stats:")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
            print(f"  Avg: {sum(latencies)/len(latencies):.2f}ms\n")

    # Failure analysis
    if failures:
        print(f"Failed queries:")
        for fail in failures[:5]:  # Show first 5
            print(f"  - Query {fail.get('query_id')}, Doc {fail.get('doc_id')}")
            if fail.get('warnings'):
                print(f"    Warnings: {fail.get('warnings')}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_logs.py <path/to/logs.jsonl>")
        print("\nExample:")
        print("  python scripts/analyze_logs.py artifacts/runs/ingest/20251015_134402_llm-judge/logs.jsonl")
        sys.exit(1)

    log_file = Path(sys.argv[1])

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    # Detect CLI type from path
    if "/ingest/" in str(log_file):
        analyze_ingest_run(log_file)
    elif "/infer/" in str(log_file):
        analyze_infer_run(log_file)
    else:
        print(f"Unknown CLI type in path: {log_file}")
        print("Supported: ingest, infer")
        sys.exit(1)


if __name__ == "__main__":
    main()
