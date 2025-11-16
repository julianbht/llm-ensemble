# Observability Quick Reference

## Start/Stop Commands

```bash
# Start observability stack
docker-compose up -d loki promtail grafana
# OR use helper script:
./docker/start-observability.sh

# Check status
docker-compose ps

# View logs
docker-compose logs -f loki
docker-compose logs -f promtail
docker-compose logs -f grafana

# Stop services
docker-compose stop loki promtail grafana

# Remove everything (including data!)
docker-compose down -v
```

## Access

- **Grafana UI**: http://localhost:3000 (admin/admin)
- **Loki API**: http://localhost:3100

## Common LogQL Queries

```logql
# All logs from a CLI
{cli="infer"}
{cli="ingest"}

# Filter by level
{level="error"}
{level=~"error|warning"}

# Filter by run
{run_name="20251116_052659_llmjudge-json"}

# Filter by model (infer only)
{model_id="gpt-oss-20b"}
{provider="openrouter"}

# Search log content
{cli="infer"} |= "error"
{cli="ingest"} |= "created"

# Aggregate queries
sum by (cli) (count_over_time({cli=~"ingest|infer"}[1h]))
sum by (level) (count_over_time({level!=""}[5m]))
```

## Available Labels

- `cli` - ingest, infer, aggregate, evaluate
- `level` - info, warning, error, debug
- `event` - Event name from structlog
- `run_type` - test, official
- `run_name` - Run identifier
- `model_id` - Model ID (infer only)
- `provider` - Provider name (infer only)

## Troubleshooting

```bash
# Check if logs are being collected
docker-compose logs promtail | grep "client/client.go"

# Check Loki health
curl http://localhost:3100/ready

# Check if Promtail can see log files
docker-compose exec promtail ls -la /var/log/llm-ensemble/

# Restart a service
docker-compose restart promtail
```

## Files Structure

```
docker/
├── loki/loki-config.yaml                    # Loki server config
├── promtail/promtail-config.yaml            # Log collection config
├── grafana/provisioning/
│   ├── datasources/loki.yaml                # Loki datasource
│   └── dashboards/
│       ├── dashboards.yaml                  # Dashboard provider
│       └── json/llm-ensemble-overview.json  # Default dashboard
├── README.md                                 # Full documentation
├── start-observability.sh                    # Quick start script
└── QUICKREF.md                               # This file
```

## Dashboard Navigation

1. Open Grafana: http://localhost:3000
2. Left sidebar → Dashboards (grid icon)
3. Select "LLM Ensemble" folder
4. Click "LLM Ensemble - Overview"

Or use **Explore** for ad-hoc queries:
1. Left sidebar → Explore (compass icon)
2. Select "Loki" datasource
3. Write LogQL query
4. Click "Run query"

## Configuration Changes

### Change log retention (default: 30 days)

Edit `docker/loki/loki-config.yaml`:
```yaml
limits_config:
  retention_period: 168h  # 7 days
```

Then: `docker-compose restart loki`

### Add new log source

Edit `docker/promtail/promtail-config.yaml` and add new `scrape_configs` entry.

Then: `docker-compose restart promtail`

## More Info

- Full guide: `docker/README.md`
- Setup summary: `docs/observability-setup.md`
- Main README: `README.md`
