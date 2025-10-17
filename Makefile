SHELL := /usr/bin/env bash
.PHONY: help install install-dev test test-ingest test-infer test-schema clean

export PYTHONUNBUFFERED=1

help:
	@echo "Available targets:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make test-ingest   - Run ingest tests only"
	@echo "  make test-infer    - Run infer tests only"
	@echo "  make test-schema   - Run schema validation tests only"
	@echo "  make clean         - Remove artifacts and cached files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest

test-ingest:
	pytest tests/ingest/

test-infer:
	pytest tests/infer/

test-schema:
	pytest -k "schema"

clean:
	rm -rf artifacts/runs/*/test_*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
