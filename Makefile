SHELL := /usr/bin/env bash
.PHONY: help install install-dev test test-ingest test-infer test-schema schemas clean
.PHONY: db db-down db-init db-logs db-status autocomplete

export PYTHONUNBUFFERED=1

help:
	@echo "Available targets:"
	@echo ""
	@echo "Installation:"
	@echo "  make install       - Install package"
	@echo "  make install-dev   - Install package with dev dependencies"
	@echo "  make autocomplete  - Install shell autocomplete for CLIs"
	@echo ""
	@echo "Database:"
	@echo "  make db            - Start PostgreSQL database (docker-compose)"
	@echo "  make db-init       - Initialize database schema (create tables, run once)"
	@echo "  make db-down       - Stop PostgreSQL database"
	@echo "  make db-status     - Check database status"
	@echo "  make db-logs       - View database logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-ingest   - Run ingest tests only"
	@echo "  make test-infer    - Run infer tests only"
	@echo "  make test-schema   - Run schema validation tests only"
	@echo ""
	@echo "Utilities:"
	@echo "  make schemas       - Generate JSON schemas from Pydantic models"
	@echo "  make clean         - Remove cached files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

autocomplete:
	@echo "Installing shell autocomplete for CLIs..."
	@echo "This will install autocomplete for: ingest, infer, aggregate, evaluate"
	@echo ""
	@if [ -d .venv ]; then \
		. .venv/bin/activate && \
		ingest --install-completion 2>/dev/null && \
		infer --install-completion 2>/dev/null && \
		aggregate --install-completion 2>/dev/null && \
		evaluate --install-completion 2>/dev/null && \
		echo "" && \
		echo "Autocomplete installed! Restart your shell or run 'source ~/.bashrc' (or ~/.zshrc)"; \
	else \
		echo "Virtual environment not found. Run 'make install-dev' first."; \
		exit 1; \
	fi

test:
	pytest

test-ingest:
	pytest tests/ingest/

test-infer:
	pytest tests/infer/

test-schema:
	pytest -k "schema"

# Database management
db:
	@echo "Starting PostgreSQL database..."
	docker-compose up -d
	@echo ""
	@echo "Database started!"
	@echo "View logs:    make db-logs"
	@echo "Check status: make db-status"

db-down:
	@echo "Stopping PostgreSQL database..."
	docker-compose down
	@echo "Database stopped (data preserved in volume)"

db-status:
	@docker-compose ps
	@echo ""
	@docker-compose exec postgres pg_isready -U llm_ensemble 2>/dev/null || echo "Database is not running. Start with 'make db-up'"

db-init:
	@echo "Initializing database schema..."
	@if [ -d .venv ]; then \
		. .venv/bin/activate && python scripts/init_db.py; \
	else \
		python3 scripts/init_db.py; \
	fi

db-logs:
	docker-compose logs -f postgres

schemas:
	@if [ -d .venv ]; then \
		. .venv/bin/activate && python scripts/generate_schemas.py; \
	else \
		python3 scripts/generate_schemas.py; \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
