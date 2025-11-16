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
    python scripts/update_model_pricing.py --model gpt-oss-20b-free
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import yaml

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_ensemble.libs.runtime.path_manager import PathManager

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_pricing(debug: bool = False) -> dict[str, dict[str, Any]]:
    """Fetch current pricing from OpenRouter API.

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
        response = requests.get(OPENROUTER_MODELS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"[ERROR] Failed to fetch OpenRouter models: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract pricing from response
    # OpenRouter returns {"data": [{"id": "...", "pricing": {"prompt": "...", "completion": "..."}}, ...]}
    pricing_map = {}

    if "data" not in data:
        print(f"[ERROR] Unexpected API response format (missing 'data' key)", file=sys.stderr)
        sys.exit(1)

    for model in data["data"]:
        model_id = model.get("id")
        pricing = model.get("pricing", {})

        if not model_id or not pricing:
            continue

        # Debug: print first model's pricing structure
        if debug and len(pricing_map) == 0:
            print(f"\n[DEBUG] Sample model: {model_id}")
            print(f"[DEBUG] Raw pricing data: {pricing}")
            print()

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
        except (ValueError, TypeError):
            print(f"[WARN] Invalid pricing for {model_id}", file=sys.stderr)
            continue

    print(f"[OK] Fetched pricing for {len(pricing_map)} models\n")
    return pricing_map


def load_yaml_preserving_structure(path: Path) -> tuple[dict, str]:
    """Load YAML and preserve raw content for partial updates.

    Returns:
        (parsed_dict, raw_content)
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    parsed = yaml.safe_load(content)
    return parsed, content


def update_config_with_pricing(
    config_path: Path,
    pricing_map: dict[str, dict[str, Any]],
    dry_run: bool = False
) -> bool:
    """Update a single model config with pricing.

    Returns:
        True if config was updated, False if skipped
    """
    # Load config
    try:
        config, raw_content = load_yaml_preserving_structure(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to load {config_path.name}: {e}", file=sys.stderr)
        return False

    # Check if it's an OpenRouter model
    if config.get("provider") != "openrouter":
        return False

    openrouter_model_id = config.get("openrouter_model_id")
    if not openrouter_model_id:
        print(f"[WARN] Skipping {config_path.name}: missing openrouter_model_id", file=sys.stderr)
        return False

    # Find pricing
    pricing = pricing_map.get(openrouter_model_id)
    if not pricing:
        print(f"[WARN] Skipping {config_path.name}: no pricing found for {openrouter_model_id}", file=sys.stderr)
        return False

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
        print(f"[DRY RUN] Would update {config_path.name} with pricing:")
        print(f"  Prompt: ${pricing['prompt']:.6f} per 1M tokens")
        print(f"  Completion: ${pricing['completion']:.6f} per 1M tokens")
    else:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f"[OK] Updated {config_path.name}")
        print(f"  Prompt: ${pricing['prompt']:.6f} per 1M tokens")
        print(f"  Completion: ${pricing['completion']:.6f} per 1M tokens")

    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update model configs with latest OpenRouter pricing"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to files"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Update only a specific model config (e.g., 'gpt-oss-20b-free')"
    )
    args = parser.parse_args()

    # Fetch pricing from OpenRouter
    pricing_map = fetch_openrouter_pricing()

    # Get model configs directory
    models_dir = PathManager.get_model_configs_dir()

    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    # Find model config files
    if args.model:
        # Single model
        config_files = [models_dir / f"{args.model}.yaml"]
        if not config_files[0].exists():
            print(f"✗ Model config not found: {config_files[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        # All YAML files
        config_files = sorted(models_dir.glob("*.yaml"))

    print(f"Processing {len(config_files)} model config(s)...\n")

    # Update each config
    updated_count = 0
    for config_path in config_files:
        if update_config_with_pricing(config_path, pricing_map, dry_run=args.dry_run):
            updated_count += 1
        print()

    # Summary
    if args.dry_run:
        print(f"[DRY RUN] Would update {updated_count}/{len(config_files)} configs")
    else:
        print(f"✓ Successfully updated {updated_count}/{len(config_files)} configs")


if __name__ == "__main__":
    main()
