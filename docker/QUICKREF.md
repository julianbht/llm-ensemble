# Observability Quick Reference

## Start/Stop

```bash
# Start
./docker/start-observability.sh

# Stop
docker compose -f docker/docker-compose.observability.yml down

# Stop and remove all data
docker compose -f docker/docker-compose.observability.yml down -v
```

## Access Points

- **Grafana**: http://localhost:3000 (admin/admin)
- **Loki API**: http://localhost:3100
- **Alloy**: http://localhost:12345

## Useful Commands

```bash
# View all logs
docker compose -f docker/docker-compose.observability.yml logs -f

# View Alloy logs (troubleshooting log collection)
docker compose -f docker/docker-compose.observability.yml logs -f alloy

# Restart Alloy (after creating new run logs)
docker compose -f docker/docker-compose.observability.yml restart alloy

# Check Loki health
curl http://localhost:3100/ready

# Query Loki directly
curl -G -s "http://localhost:3100/loki/api/v1/query" --data-urlencode 'query={job="llm-logs"}' | jq
```

## Common LogQL Queries

```logql
# All logs
{job="llm-logs"}

# Infer CLI only
{job="llm-logs", cli="infer"}

# Errors only
{job="llm-logs"} | json | level="error"

# Specific run
{job="llm-logs", run_name="20251116_120000_gpt-oss"}

# Search text
{job="llm-logs"} |= "OpenRouter"
```

## Grafana Dashboard

The **LLM Ensemble Overview** dashboard is auto-provisioned and includes:

- Log distribution by CLI and level
- Error/warning counts
- Log rate time series
- Live log stream (auto-refresh every 5s)
- Dedicated infer CLI panel for long-running jobs

## Typical Workflow

1. Start observability stack: `./docker/start-observability.sh`
2. Open Grafana: http://localhost:3000
3. Run your CLI with logging enabled
4. Monitor in real-time or analyze after completion
5. Query historical data from past runs

## Troubleshooting

**No logs in Grafana?**
- Ensure CLIs are writing logs (`save_logs: true` in logging config)
- Restart Alloy: `docker compose -f docker/docker-compose.observability.yml restart alloy`
- Check Alloy logs: `docker compose -f docker/docker-compose.observability.yml logs alloy`

**Dashboard not loading?**
- Verify Loki datasource in Grafana (Configuration → Data Sources)
- Check Grafana logs: `docker compose -f docker/docker-compose.observability.yml logs grafana`

**Want to clear old logs?**
- Loki automatically retains logs for 7 days
- To clear immediately: `docker compose -f docker/docker-compose.observability.yml down -v`
