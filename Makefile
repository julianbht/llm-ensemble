SHELL := /usr/bin/env bash
.PHONY: help install install-dev test test-ingest test-infer test-schema schemas clean clean-test-runs
.PHONY: db db-down db-init db-logs db-status autocomplete update-pricing
.PHONY: observability observability-down observability-reset observability-logs observability-status infra infra-down

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
	@echo "Observability (Grafana/Loki/Alloy):"
	@echo "  make observability        - Start observability stack (Grafana, Loki, Alloy)"
	@echo "  make observability-down   - Stop observability stack"
	@echo "  make observability-reset  - Reset observability stack (clear all data and restart)"
	@echo "  make observability-status - Check observability services status"
	@echo "  make observability-logs   - View observability logs (all services)"
	@echo ""
	@echo "Infrastructure (All services):"
	@echo "  make infra         - Start all infrastructure (DB + Observability)"
	@echo "  make infra-down    - Stop all infrastructure"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-ingest   - Run ingest tests only"
	@echo "  make test-infer    - Run infer tests only"
	@echo "  make test-schema   - Run schema validation tests only"
	@echo ""
	@echo "Utilities:"
	@echo "  make schemas          - Generate JSON schemas from Pydantic models"
	@echo "  make update-pricing   - Update model configs with latest OpenRouter pricing"
	@echo "  make clean            - Remove cached files"
	@echo "  make clean-test-runs  - Remove all test run artifacts (keeps official runs)"

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
	elif [ -d venv ]; then \
		. venv/bin/activate && \
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
	elif [ -d venv ]; then \
		. venv/bin/activate && python scripts/init_db.py; \
	else \
		python3 scripts/init_db.py; \
	fi

db-logs:
	docker-compose logs -f postgres

# Observability management (Grafana, Loki, Alloy)
observability:
	@echo "Starting observability stack (Grafana, Loki, Alloy)..."
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml up -d
	@echo ""
	@echo "Waiting for services to start..."
	@sleep 3
	@echo ""
	@echo "Grafana Dashboard:"
	@echo "   URL:      http://localhost:3000"
	@echo "   Username: admin"
	@echo "   Password: admin"
	@echo ""
	@echo "Loki API:    http://localhost:3100"
	@echo "Alloy UI:    http://localhost:12345"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Open Grafana at http://localhost:3000"
	@echo "  2. Run a CLI to generate logs (e.g., 'infer --model gpt-oss-20b --prompt thomas-et-al-prompt --io json --input samples.json')"
	@echo "  3. View logs in the 'LLM Ensemble Overview' dashboard"
	@echo ""
	@echo "Status: make observability-status"
	@echo "Logs:   make observability-logs"

observability-down:
	@echo "Stopping observability stack..."
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml down
	@echo "Observability stack stopped (data preserved in volumes)"

observability-reset:
	@echo "Resetting observability stack (clearing all data)..."
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml down -v
	@echo "All observability data cleared (Loki logs, Grafana dashboards)"
	@echo "Restarting with fresh volumes..."
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml up -d
	@sleep 3
	@echo ""
	@echo "Observability stack reset complete!"
	@echo "Grafana: http://localhost:3000"

observability-status:
	@echo "Observability Services Status:"
	@echo "=============================="
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml ps

observability-logs:
	@echo "Streaming logs from observability services (Ctrl+C to exit)..."
	@echo "=============================="
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml logs -f

# Infrastructure management (all services)
infra:
	@echo "Starting all infrastructure services..."
	@docker-compose up -d
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml up -d
	@echo ""
	@echo "All services started!"
	@echo ""
	@echo "PostgreSQL Database:"
	@echo "  Port:     5432"
	@echo "  User:     llm_ensemble"
	@echo "  Database: llm_ensemble"
	@echo ""
	@echo "Grafana Dashboard:"
	@echo "  URL:      http://localhost:3000"
	@echo "  Username: admin"
	@echo "  Password: admin"
	@echo ""
	@echo "Initialize DB schema: make db-init"
	@echo "Check status:         make db-status && make observability-status"

infra-down:
	@echo "Stopping all infrastructure services..."
	@docker compose -p llm-ensemble down
	@docker compose -p llm-ensemble-observability -f docker/docker-compose.observability.yml down
	@echo "All services stopped (data preserved in volumes)"

schemas:
	@if [ -d .venv ]; then \
		. .venv/bin/activate && python scripts/generate_schemas.py; \
	elif [ -d venv ]; then \
		. venv/bin/activate && python scripts/generate_schemas.py; \
	else \
		python3 scripts/generate_schemas.py; \
	fi

update-pricing:
	@if [ -d .venv ]; then \
		. .venv/bin/activate && python scripts/update_model_pricing.py $(ARGS); \
	elif [ -d venv ]; then \
		. venv/bin/activate && python scripts/update_model_pricing.py $(ARGS); \
	else \
		python3 scripts/update_model_pricing.py $(ARGS); \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

clean-test-runs:
	@echo "Cleaning all test run artifacts..."
	@echo ""
	@if [ -d artifacts/runs/ingest/test ]; then \
		echo "Removing ingest test runs..."; \
		rm -rf artifacts/runs/ingest/test/*; \
		echo "  Removed artifacts/runs/ingest/test/*"; \
	fi
	@if [ -d artifacts/runs/infer/test ]; then \
		echo "Removing infer test runs..."; \
		rm -rf artifacts/runs/infer/test/*; \
		echo "  Removed artifacts/runs/infer/test/*"; \
	fi
	@if [ -d artifacts/runs/aggregate/test ]; then \
		echo "Removing aggregate test runs..."; \
		rm -rf artifacts/runs/aggregate/test/*; \
		echo "  Removed artifacts/runs/aggregate/test/*"; \
	fi
	@if [ -d artifacts/runs/evaluate/test ]; then \
		echo "Removing evaluate test runs..."; \
		rm -rf artifacts/runs/evaluate/test/*; \
		echo "  Removed artifacts/runs/evaluate/test/*"; \
	fi
	@echo ""
	@echo "Test runs cleaned! Official runs preserved."
	@echo "Note: Official runs in artifacts/runs/*/official/ were not touched."
