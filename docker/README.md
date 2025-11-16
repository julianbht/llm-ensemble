# Observability Setup for LLM Ensemble

This directory contains the Grafana observability stack configuration for monitoring LLM Ensemble CLI logs.

## Architecture

The observability stack consists of three components:

1. **Loki** - Log aggregation system that stores and indexes logs
2. **Promtail** - Log shipper that reads `run.log` files and sends them to Loki
3. **Grafana** - Visualization platform for querying and displaying logs

```
artifacts/runs/
  ├── ingest/test/*/run.log  ─┐
  ├── infer/test/*/run.log   ─┤
  ├── aggregate/test/*/run.log─┼─→ Promtail ──→ Loki ──→ Grafana
  └── evaluate/test/*/run.log ─┘
```

## Quick Start

### 1. Start the observability stack

```bash
docker-compose up -d loki promtail grafana
```

### 2. Access Grafana

Open http://localhost:3000 in your browser

**Credentials:**
- Username: `admin`
- Password: `admin` (you'll be prompted to change this on first login)

### 3. View logs

The "LLM Ensemble - Overview" dashboard is automatically provisioned and shows:
- All CLI logs in real-time
- Log distribution by CLI (ingest/infer/aggregate/evaluate)
- Log distribution by level (info/warning/error)
- Filtered views for specific CLIs
- Errors and warnings panel

### 4. Run a CLI to generate logs

```bash
# Example: run ingest
ingest --io llm_judge_ingest --limit 10

# Wait a few seconds, then check Grafana
```

## Log Exploration

### Using LogQL (Loki Query Language)

In Grafana's Explore view, you can write LogQL queries:

```logql
# All logs from infer CLI
{cli="infer"}

# All errors across all CLIs
{level="error"}

# Logs from a specific run
{run_name="20251116_052659_llmjudge-json"}

# Infer logs with model info
{cli="infer", model_id!=""}

# Warning or error logs
{level=~"warning|error"}

# Search for specific text in logs
{cli="ingest"} |= "error"

# Count errors by CLI in last hour
sum by (cli) (count_over_time({level="error"}[1h]))
```

### Available Labels

Promtail extracts these labels from your structlog JSON logs:

- `cli` - CLI name (ingest, infer, aggregate, evaluate)
- `level` - Log level (info, warning, error, debug)
- `event` - Event name from structlog
- `run_type` - Run type (test, official)
- `run_name` - Run identifier
- `model_id` - Model ID (infer CLI only)
- `provider` - Provider name (infer CLI only)

All other JSON fields remain searchable in the log content.

## Configuration Files

- **`loki/loki-config.yaml`** - Loki server configuration
  - 30-day log retention
  - Filesystem-based storage (for local dev)
  - No authentication (local only)

- **`promtail/promtail-config.yaml`** - Log collection configuration
  - Watches `artifacts/runs/*/*/run.log` files
  - Parses JSON logs and extracts labels
  - Separate scrape configs for each CLI

- **`grafana/provisioning/`** - Auto-provisioning configuration
  - `datasources/` - Loki datasource
  - `dashboards/` - Default dashboards

## Maintenance

### View service logs

```bash
docker-compose logs -f loki      # Loki server logs
docker-compose logs -f promtail  # Promtail collection logs
docker-compose logs -f grafana   # Grafana server logs
```

### Check service health

```bash
docker-compose ps
```

All services should show "healthy" status.

### Reset all log data

```bash
docker-compose down -v  # WARNING: Deletes all data including Grafana dashboards
docker-compose up -d
```

### Stop observability stack

```bash
docker-compose stop loki promtail grafana
```

## Troubleshooting

### Logs not appearing in Grafana

1. Check Promtail is running and healthy:
   ```bash
   docker-compose logs promtail
   ```

2. Verify log files exist:
   ```bash
   ls -la artifacts/runs/*/test/*/run.log
   ```

3. Check Promtail can read the files (permission issues):
   ```bash
   docker-compose exec promtail ls -la /var/log/llm-ensemble/ingest/test/
   ```

4. Verify Loki is receiving logs:
   ```bash
   curl http://localhost:3100/ready
   curl http://localhost:3100/metrics | grep promtail
   ```

### Grafana shows "No datasource"

The Loki datasource should be auto-provisioned. If missing:

1. Check provisioning config is mounted:
   ```bash
   docker-compose exec grafana ls -la /etc/grafana/provisioning/datasources/
   ```

2. Manually add Loki datasource in Grafana UI:
   - Go to Configuration → Data Sources → Add data source
   - Select Loki
   - URL: `http://loki:3100`
   - Save & Test

### High disk usage

Loki stores logs for 30 days by default. To reduce retention:

1. Edit `docker/loki/loki-config.yaml`
2. Change `retention_period: 720h` to a lower value (e.g., `168h` for 7 days)
3. Restart Loki: `docker-compose restart loki`

## Production Considerations

This setup is optimized for local development. For production:

1. **Authentication** - Enable Loki authentication and use secrets for credentials
2. **Storage** - Use object storage (S3, GCS) instead of local filesystem
3. **Retention** - Adjust based on compliance and storage costs
4. **Resources** - Increase memory limits for Loki based on log volume
5. **Monitoring** - Add Prometheus to monitor the observability stack itself
6. **Backups** - Backup Grafana configuration and dashboards
7. **TLS** - Use HTTPS for Grafana and internal service communication

## References

- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Promtail Configuration](https://grafana.com/docs/loki/latest/clients/promtail/configuration/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/dashboards/)
