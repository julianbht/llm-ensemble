#!/bin/bash
# Quick start script for LLM Ensemble observability stack
# This script starts the Grafana/Loki/Promtail stack and provides helpful info

set -e

echo "🚀 Starting LLM Ensemble Observability Stack..."
echo ""

# Start services
docker-compose up -d loki promtail grafana

echo ""
echo "⏳ Waiting for services to become healthy..."
sleep 5

# Check health
if docker-compose ps | grep -q "healthy"; then
    echo "✅ Services are starting up!"
else
    echo "⚠️  Services may still be initializing. Check status with: docker-compose ps"
fi

echo ""
echo "📊 Grafana Dashboard:"
echo "   URL:      http://localhost:3000"
echo "   Username: admin"
echo "   Password: admin"
echo ""
echo "📝 Loki API:"
echo "   URL:      http://localhost:3100"
echo ""
echo "🔍 Next steps:"
echo "   1. Open Grafana at http://localhost:3000"
echo "   2. Navigate to 'Dashboards' → 'LLM Ensemble' → 'Overview'"
echo "   3. Run a CLI to generate logs:"
echo "      ingest --io llm_judge_ingest --limit 10"
echo "   4. Refresh the dashboard to see logs appear"
echo ""
echo "📖 Full documentation: docker/README.md"
echo ""
echo "To stop services: docker-compose stop loki promtail grafana"
echo "To view logs:     docker-compose logs -f <service>"
