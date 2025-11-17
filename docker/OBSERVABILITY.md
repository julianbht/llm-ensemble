# Observability Stack for LLM Ensemble

Minimal Docker-based observability setup using Grafana, Loki, and Grafana Alloy for monitoring LLM Ensemble CLIs.

## Overview

This setup provides:

1. **Log Analysis**: View and search logs from all 4 CLIs (ingest, infer, aggregate, evaluate)
2. **Live Monitoring**: Real-time dashboard for monitoring long-running infer CLI jobs
3. **Historical Analysis**: Query and analyze past runs stored in Loki

## Components

- **Loki**: Log aggregation and storage (7-day retention)
- **Grafana Alloy**: Log collection agent that tails `artifacts/runs/**/run.log` files
- **Grafana**: Visualization and dashboards

## Quick Start

### 1. Start the Stack

```bash
./docker/start-observability.sh
```

Or manually:

```bash
cd docker
docker compose -f docker-compose.observability.yml up -d
```

### 2. Access Grafana

Open http://localhost:3000 in your browser.

- Username: `admin`
- Password: `admin`

The **LLM Ensemble Overview** dashboard is provisioned automatically.

### 3. Run Your CLIs

Ensure your CLIs are configured to save logs. The logging config should have `save_logs: true`:

```yaml
# configs/logging/standard.yaml
pretty_print: false  # Use compact JSON for better Loki ingestion
save_logs: true
console_level: INFO
file_level: DEBUG
```

Run any CLI as usual:

```bash
infer --model gpt-oss-20b --prompt thomas-et-al-prompt --io json --input samples.json
```

Logs will be written to `artifacts/runs/infer/<run_name>/run.log` and automatically picked up by Grafana Alloy.

## Dashboard Features

The **LLM Ensemble Overview** dashboard provides:

### Overview Panels
- **Logs by CLI**: Pie chart showing distribution of logs across CLIs
- **Logs by Level**: Breakdown of log levels (info, warning, error)
- **Total Errors/Warnings**: Quick stats on issues

### Time Series
- **Log Rate by CLI**: Monitor activity over time

### Log Panels
- **All Logs (Live)**: Real-time stream of all logs with 5-second refresh
- **Errors & Warnings**: Filtered view of issues
- **Infer CLI Logs**: Dedicated panel for monitoring long-running inference jobs

### Using the Dashboard

1. **Live Monitoring**: Set time range to "Last 30 minutes" and enable auto-refresh (5s)
2. **Historical Analysis**: Adjust time range to view past runs
3. **Filter by CLI**: Click on CLI names in the legend to filter
4. **Explore Logs**: Click on any log entry to see full JSON details

## Querying Logs in Grafana

Grafana uses LogQL (Loki Query Language) to query logs. Here are some useful queries:

### Basic Queries

```logql
# All logs
{job="llm-logs"}

# Filter by CLI
{job="llm-logs", cli="infer"}

# Filter by run name
{job="llm-logs", run_name="20251116_120000_gpt-oss"}

# Filter by level
{job="llm-logs"} | json | level="error"

# Filter by event
{job="llm-logs"} | json | event="model_inference_started"
```

### Advanced Queries

```logql
# Count errors per CLI
sum by(cli) (count_over_time({job="llm-logs"} | json | level="error" [5m]))

# Find slow inferences (example - adjust based on your log structure)
{job="llm-logs", cli="infer"} | json | latency_ms > 5000

# Search for specific text in logs
{job="llm-logs"} |= "OpenRouter"

# Combine filters
{job="llm-logs", cli="infer"} | json | level="error" | line_format "{{.event}}: {{.error}}"
```

## Managing the Stack

### View Logs

```bash
# All services
docker compose -f docker/docker-compose.observability.yml logs -f

# Specific service
docker compose -f docker/docker-compose.observability.yml logs -f alloy
```

### Stop the Stack

```bash
docker compose -f docker/docker-compose.observability.yml down
```

### Stop and Remove Data

```bash
docker compose -f docker/docker-compose.observability.yml down -v
```

### Restart a Service

```bash
docker compose -f docker/docker-compose.observability.yml restart alloy
```

## Configuration Files

- `docker-compose.observability.yml`: Main orchestration file
- `loki/loki-config.yaml`: Loki configuration (retention, limits)
- `alloy/alloy-config.alloy`: Alloy log collection configuration
- `grafana/provisioning/`: Grafana datasources and dashboards

### Customization

#### Change Log Retention

Edit `docker/loki/loki-config.yaml`:

```yaml
limits_config:
  retention_period: 168h  # Change to desired hours (default: 7 days)
```

Restart Loki:

```bash
docker compose -f docker/docker-compose.observability.yml restart loki
```

#### Modify Log Collection

Edit `docker/alloy/alloy-config.alloy` to change which files are tailed or add additional labels.

Restart Alloy:

```bash
docker compose -f docker/docker-compose.observability.yml restart alloy
```

#### Customize Dashboards

1. Edit dashboards in Grafana UI (dashboards are editable)
2. Export JSON from Grafana
3. Save to `docker/grafana/provisioning/dashboards/json/`
4. Restart Grafana to reload

## Troubleshooting

### Services Not Starting (Permission Denied)

If containers are restarting with "permission denied" errors:

```bash
# Fix config file permissions
chmod 644 docker/loki/loki-config.yaml
chmod 644 docker/alloy/alloy-config.alloy
chmod -R 755 docker/grafana/provisioning
find docker/grafana/provisioning -type f -exec chmod 644 {} \;

# Restart services
docker compose -f docker/docker-compose.observability.yml restart
```

This can happen if the config files are created with restrictive permissions (mode 600). Docker containers run as specific users (UID 10001) and need read access.

### No Logs Appearing in Grafana

1. Check if CLIs are writing logs:
   ```bash
   ls -la artifacts/runs/*/test/*/run.log
   ```

2. Check if Alloy is discovering files:
   ```bash
   docker compose -f docker/docker-compose.observability.yml logs alloy | grep "found"
   ```

3. Verify Alloy can reach Loki:
   ```bash
   docker compose -f docker/docker-compose.observability.yml exec alloy curl http://loki:3100/ready
   ```

4. Check Loki ingestion:
   ```bash
   curl http://localhost:3100/loki/api/v1/label
   ```

### Dashboard Not Loading

1. Check if datasource is configured:
   - Go to Grafana → Configuration → Data Sources
   - Verify "Loki" datasource exists and is reachable

2. Check dashboard provisioning:
   ```bash
   docker compose -f docker/docker-compose.observability.yml logs grafana | grep "dashboard"
   ```

### Alloy Not Tailing Files

Alloy requires files to exist when it starts. If you start Alloy before running any CLIs, it won't discover files until restarted.

Solution: Restart Alloy after creating new log files:

```bash
docker compose -f docker/docker-compose.observability.yml restart alloy
```

Alternatively, configure your first CLI run, then start the observability stack.

## Architecture

```
CLI (ingest/infer/aggregate/evaluate)
  ↓ writes JSON logs
artifacts/runs/<cli>/<run_name>/run.log
  ↓ tailed by
Grafana Alloy
  ↓ ships logs to
Loki (storage & indexing)
  ↓ queried by
Grafana (visualization)
```

## Next Steps

- **Custom Dashboards**: Create dashboards for specific use cases (model comparison, cost tracking, etc.)
- **Alerting**: Configure Grafana alerts for errors or performance issues
- **Metrics**: Add Prometheus for system metrics (CPU, memory, disk)
- **Longer Retention**: Configure Loki to use object storage (S3, GCS) for longer retention
- **Production Setup**: Add authentication, HTTPS, and proper secret management

## Resources

- [Grafana Documentation](https://grafana.com/docs/grafana/latest/)
- [Loki Documentation](https://grafana.com/docs/loki/latest/)
- [Grafana Alloy Documentation](https://grafana.com/docs/alloy/latest/)
- [LogQL Query Language](https://grafana.com/docs/loki/latest/logql/)
