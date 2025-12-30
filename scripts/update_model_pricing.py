#!/usr/bin/env python3
"""Update model configs with latest pricing from OpenRouter API.

This script:
1. Queries OpenRouter's /api/v1/models endpoint for current pricing
2. Finds all model configs in configs/models/
3. Updates OpenRouter model configs with pricing information
4. Preserves existing config structure and comments

Usage:
    python scripts/update_model_pricing.py
    python scripts/update_model_pricing.py --dry-run
    python scripts/update_model_pricing.py --model gemma-3n-e2b-it-free
    python scripts/update_model_pricing.py --debug
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import requests
import typer
import yaml

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_ensemble.libs.runtime.path_manager import PathManager

app = typer.Typer(
    add_completion=True,
    help="Update model configs with latest OpenRouter pricing",
    pretty_exceptions_enable=False,
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_pricing(debug: bool = False) -> dict[str, dict[str, float]]:
    """Fetch current pricing from OpenRouter API.

    Args:
        debug: Whether to print debug information

    Returns:
        Dict mapping model IDs to pricing info: {
            "openai/gpt-4": {
                "prompt": 0.03,      # per 1M tokens
                "completion": 0.06,  # per 1M tokens
            }
        }
    """
    print(f"Fetching pricing from {OPENROUTER_MODELS_URL}...")

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error: API request failed: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Extract pricing from response
    # OpenRouter returns {"data": [{"id": "...", "pricing": {"prompt": "...", "completion": "..."}}, ...]}
    pricing_map = {}

    if "data" not in data:
        print("Error: Invalid API response (missing 'data' key)", file=sys.stderr)
        raise typer.Exit(code=1)

    for model in data["data"]:
        model_id = model.get("id")
        pricing = model.get("pricing", {})

        if not model_id or not pricing:
            continue

        # Debug: log first model's pricing structure
        if debug and len(pricing_map) == 0:
            print(f"\nDebug: Sample pricing structure for {model_id}:")
            print(f"  Raw pricing: {pricing}")

        # Convert string prices to floats and scale to per-1M tokens
        # OpenRouter returns prices as strings in dollars per token
        try:
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))

            # Scale to per-1M tokens (API returns per-token prices)
            pricing_map[model_id] = {
                "prompt": prompt_price * 1_000_000,
                "completion": completion_price * 1_000_000,
            }
        except (ValueError, TypeError) as e:
            if debug:
                print(f"Warning: Could not parse pricing for {model_id}: {e}", file=sys.stderr)
            continue

    print(f"Fetched pricing for {len(pricing_map)} models\n")
    return pricing_map


def load_yaml_preserving_structure(path: Path) -> tuple[dict, str]:
    """Load YAML and preserve raw content for partial updates.

    Args:
        path: Path to YAML file

    Returns:
        Tuple of (parsed_dict, raw_content)
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = yaml.safe_load(content)
    return parsed, content


def update_config_with_pricing(
    config_path: Path,
    pricing_map: dict[str, dict[str, float]],
    dry_run: bool,
    debug: bool = False,
) -> str:
    """Update a single model config with pricing.

    Args:
        config_path: Path to model config file
        pricing_map: Mapping of model IDs to pricing info
        dry_run: Whether to preview changes without writing
        debug: Whether to print debug information

    Returns:
        "updated" if config was updated, "skipped" if not applicable
    """
    # Load config
    try:
        config, raw_content = load_yaml_preserving_structure(config_path)
    except Exception as e:
        print(f"Error loading {config_path.name}: {e}", file=sys.stderr)
        return "skipped"

    # Check if it's an OpenRouter model (model_id contains "/" like "openai/gpt-4")
    model_id = config.get("model_id")
    if not model_id or "/" not in model_id:
        if debug:
            print(f"Skipped {config_path.name}: not an OpenRouter model (model_id={model_id})")
        return "skipped"

    # Find pricing (use model_id directly)
    pricing = pricing_map.get(model_id)
    if not pricing:
        print(f"Warning: No pricing available for {config_path.name} (model_id={model_id})", file=sys.stderr)
        return "skipped"

    # Build pricing section
    timestamp = datetime.now(timezone.utc).isoformat()
    pricing_section = f"""
# Pricing information (auto-updated from OpenRouter API)
pricing:
  prompt_cost_per_1m_tokens: {pricing['prompt']}
  completion_cost_per_1m_tokens: {pricing['completion']}
  last_updated: "{timestamp}"
"""

    # Check if pricing section already exists
    if "pricing:" in raw_content:
        # Remove old pricing section (between "pricing:" and next top-level key or EOF)
        lines = raw_content.split("\n")
        new_lines = []
        in_pricing = False

        for line in lines:
            # Start of pricing section
            if line.strip().startswith("pricing:") or (
                line.strip().startswith("# Pricing information")
            ):
                in_pricing = True
                continue

            # End of pricing section (next top-level key, or indented line becomes non-indented)
            if in_pricing and line and not line.startswith(" ") and not line.startswith("#"):
                in_pricing = False

            if not in_pricing:
                new_lines.append(line)

        updated_content = "\n".join(new_lines).rstrip() + "\n" + pricing_section.strip() + "\n"
    else:
        # Append pricing section at the end
        updated_content = raw_content.rstrip() + "\n\n" + pricing_section.strip() + "\n"

    # Write back
    if dry_run:
        print(f"Would update {config_path.name}: "
              f"prompt=${pricing['prompt']:.6f}/1M, "
              f"completion=${pricing['completion']:.6f}/1M")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"Updated {config_path.name}: "
              f"prompt=${pricing['prompt']:.6f}/1M, "
              f"completion=${pricing['completion']:.6f}/1M")

    return "updated"


@app.command()
def update_pricing(
    model_name: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help="Update only a specific model config (e.g., 'gemma-3n-e2b-it-free'). If not specified, updates all OpenRouter models.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview changes without writing to files",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Show debug information including API response samples",
        ),
    ] = False,
) -> None:
    """Update model configs with latest pricing from OpenRouter API.

    This command:
    1. Fetches current pricing from OpenRouter's /api/v1/models endpoint
    2. Finds all OpenRouter model configs in configs/models/
    3. Updates pricing fields while preserving existing config structure
    4. Adds timestamp to track when pricing was last updated

    Examples:

        # Preview updates for all models
        python scripts/update_model_pricing.py --dry-run

        # Update all OpenRouter models
        python scripts/update_model_pricing.py

        # Update specific model
        python scripts/update_model_pricing.py --model gemma-3n-e2b-it-free

        # Debug mode to see API response structure
        python scripts/update_model_pricing.py --dry-run --debug
    """
    print(f"Update Model Pricing")
    print(f"===================")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"Target: {model_name or 'all models'}\n")

    # Fetch pricing from OpenRouter
    pricing_map = fetch_openrouter_pricing(debug)

    # Get model configs directory
    models_dir = PathManager.get_model_configs_dir()

    if not models_dir.exists():
        print(f"Error: Models directory not found: {models_dir}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Find model config files
    if model_name:
        # Single model
        config_files = [models_dir / f"{model_name}.yaml"]
        if not config_files[0].exists():
            print(f"Error: Model config not found: {config_files[0]}", file=sys.stderr)
            raise typer.Exit(code=1)
    else:
        # All YAML files
        config_files = sorted(models_dir.glob("*.yaml"))

    print(f"Processing {len(config_files)} config file(s) from {models_dir.relative_to(PathManager.get_project_root())}\n")

    # Update each config
    updated_count = 0
    skipped_count = 0

    for config_path in config_files:
        result = update_config_with_pricing(
            config_path, pricing_map, dry_run, debug
        )
        if result == "updated":
            updated_count += 1
        elif result == "skipped":
            skipped_count += 1

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Total configs: {len(config_files)}")
    print(f"  Updated: {updated_count}")
    print(f"  Skipped: {skipped_count}")
    if dry_run:
        print(f"\nRun without --dry-run to apply changes")


if __name__ == "__main__":
    app()
