# Grafana Observability Setup - Summary

This document provides a complete overview of the Grafana/Loki/Promtail observability stack added to the LLM Ensemble project.

## What Was Added

### 1. Docker Compose Services (docker-compose.yml)

Added three new services to the existing docker-compose.yml:

- **Loki** (port 3100) - Log aggregation and storage backend
- **Promtail** - Log collector that ships logs from `artifacts/runs/` to Loki
- **Grafana** (port 3000) - Visualization and dashboarding UI

All services are connected via a Docker network and include health checks for reliable startup.

### 2. Configuration Files

Created infrastructure configuration in `/docker`:

```
docker/
├── loki/
│   └── loki-config.yaml          # Loki server config (30-day retention)
├── promtail/
│   └── promtail-config.yaml      # Log collection config (watches run.log files)
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── loki.yaml         # Auto-provision Loki datasource
│       └── dashboards/
│           ├── dashboards.yaml   # Auto-provision dashboards
│           └── json/
│               └── llm-ensemble-overview.json  # Default dashboard
├── README.md                      # Comprehensive observability guide
└── start-observability.sh         # Quick start script
```

### 3. Documentation Updates

- Updated main `README.md` with observability section
- Created detailed `docker/README.md` with setup, usage, and troubleshooting

## How It Works

### Data Flow

```
Your CLIs write structlog JSON → artifacts/runs/*/*/run.log
                                           ↓
                                    Promtail watches files
                                           ↓
                                  Parses JSON, extracts labels
                                           ↓
                                    Ships to Loki
                                           ↓
                                   Loki indexes & stores
                                           ↓
                                 Grafana queries & visualizes
```

### Log Collection

Promtail is configured to:
- Watch all `run.log` files in `artifacts/runs/`
- Parse the JSON structure from structlog
- Extract labels: `cli`, `level`, `event`, `run_type`, `run_name`, `model_id`, `provider`
- Ship to Loki with proper timestamps

### Labels Available for Filtering

From your structlog JSON logs, these labels are automatically extracted:

- `cli` - Which CLI generated the log (ingest/infer/aggregate/evaluate)
- `level` - Log level (info/warning/error/debug)
- `event` - Event name from structlog
- `run_type` - test or official
- `run_name` - Run identifier (e.g., "20251116_052659_llmjudge-json")
- `model_id` - Model ID (infer CLI only)
- `provider` - Provider name (infer CLI only)

All other JSON fields remain searchable in the log content.

## Quick Start

### Start the stack

```bash
# Option 1: Use helper script
./docker/start-observability.sh

# Option 2: Docker Compose directly
docker-compose up -d loki promtail grafana
```

### Access Grafana

1. Open http://localhost:3000
2. Login with `admin` / `admin`
3. Go to Dashboards → LLM Ensemble → Overview

### Generate some logs

```bash
# Run any CLI to generate logs
ingest --io llm_judge_ingest --limit 10

# Or run infer
infer --model gpt-oss-20b --prompt thomas-et-al-prompt --io json --input <path>
```

Logs appear in Grafana within seconds!

## Example Queries

In Grafana's Explore view, try these LogQL queries:

```logql
# All logs from infer CLI
{cli="infer"}

# All errors
{level="error"}

# Logs from a specific run
{run_name="20251116_052659_llmjudge-json"}

# Infer logs with specific model
{cli="infer", model_id="gpt-oss-20b"}

# Search for text in logs
{cli="ingest"} |= "created"

# Count logs by CLI over time
sum by (cli) (count_over_time({cli=~"ingest|infer"}[5m]))
```

## Default Dashboard

The "LLM Ensemble - Overview" dashboard includes:

1. **All CLI Logs** - Real-time log stream from all CLIs
2. **Log Entries by CLI** - Pie chart showing distribution
3. **Log Entries by Level** - Pie chart showing info/warning/error distribution
4. **Infer CLI Logs** - Filtered view of inference logs
5. **Errors and Warnings** - All problems across CLIs

You can customize this dashboard or create new ones through the Grafana UI.

## Benefits

1. **Centralized Logging** - All CLI runs in one place
2. **Real-time Monitoring** - See logs as they're written
3. **Powerful Filtering** - Query by CLI, level, run, model, etc.
4. **Time-based Analysis** - See what happened when
5. **JSON-aware** - All structlog fields are searchable
6. **Persistent Storage** - Logs kept for 30 days (configurable)
7. **No Code Changes** - Works with existing structlog JSON logs

## Architecture Decisions

### Why `/docker` instead of `/configs`?

Infrastructure configuration lives in `/docker` because:
- These configs are consumed by Docker services, not your Python application
- Separates application config (models, prompts) from infrastructure (Loki, Grafana)
- Follows 12-factor app principle of environment-specific configuration
- Makes it clear these are deployment concerns, not application logic

### Why Loki instead of Elasticsearch?

- **Lightweight** - Lower resource requirements for local development
- **Cost-effective** - Labels-first approach reduces storage needs
- **Native Grafana integration** - Best-in-class visualization
- **LogQL** - Powerful query language similar to PromQL
- **Designed for logs** - Purpose-built, not repurposed from search engine

### Why Promtail?

- **Native Loki integration** - Built by the same team
- **File watching** - Perfect for our file-based log output
- **JSON parsing** - First-class support for structured logs
- **Label extraction** - Flexible pipeline for metadata

## Production Considerations

This setup is optimized for local development. For production, consider:

1. **Authentication** - Enable auth in Loki, use secrets for Grafana
2. **Storage** - Use S3/GCS instead of local filesystem
3. **Retention** - Adjust based on compliance requirements
4. **Resources** - Scale Loki replicas based on log volume
5. **Monitoring** - Add Prometheus to monitor the stack itself
6. **Backups** - Backup Grafana dashboards and Loki data
7. **TLS** - Use HTTPS for all communications

## Troubleshooting

### Logs not appearing?

1. Check Promtail logs: `docker-compose logs promtail`
2. Verify files exist: `ls artifacts/runs/*/test/*/run.log`
3. Check Loki health: `curl http://localhost:3100/ready`

### Grafana connection issues?

1. Verify Loki is running: `docker-compose ps loki`
2. Check datasource config: Grafana UI → Configuration → Data Sources
3. Test connection in Grafana UI

### High disk usage?

1. Edit `docker/loki/loki-config.yaml`
2. Reduce `retention_period` (default 720h = 30 days)
3. Restart: `docker-compose restart loki`

## References

- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Promtail Configuration](https://grafana.com/docs/loki/latest/clients/promtail/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)

## Next Steps

1. **Start the stack** - Run `./docker/start-observability.sh`
2. **Explore the UI** - Open Grafana and familiarize yourself with the interface
3. **Run some CLIs** - Generate logs from ingest and infer
4. **Learn LogQL** - Try different queries in Explore view
5. **Customize dashboards** - Add panels for your specific needs
6. **Set up alerts** (optional) - Configure Grafana alerts for errors

Enjoy your new observability superpowers! 🚀
