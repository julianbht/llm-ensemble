#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting LLM Ensemble Observability Stack..."
echo ""
echo "This will start:"
echo "  - Loki (log aggregation) on http://localhost:3100"
echo "  - Grafana Alloy (log collection)"
echo "  - Grafana (dashboards) on http://localhost:3000"
echo ""

cd "$SCRIPT_DIR"

docker compose -f docker-compose.observability.yml up -d

echo ""
echo "Observability stack started successfully!"
echo ""
echo "Access Grafana at: http://localhost:3000"
echo "  Username: admin"
echo "  Password: admin"
echo ""
echo "The 'LLM Ensemble Overview' dashboard should be available automatically."
echo ""
echo "To stop the stack, run:"
echo "  docker compose -f docker/docker-compose.observability.yml down"
echo ""
echo "To view logs:"
echo "  docker compose -f docker/docker-compose.observability.yml logs -f"
