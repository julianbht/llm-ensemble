#!/usr/bin/env python3
"""List OpenRouter models with pricing information.

This script:
1. Queries OpenRouter's /api/v1/models endpoint for current pricing
2. Displays models with their pricing sorted by cost
3. Supports filtering by free/paid and sorting by different cost metrics

Usage:
    python scripts/list_openrouter_models.py --paid-only
    python scripts/list_openrouter_models.py --paid-only --limit 10
    python scripts/list_openrouter_models.py --free-only
    python scripts/list_openrouter_models.py --sort-by completion
"""

import sys
from typing import Annotated, Optional

import requests
import typer

app = typer.Typer(
    add_completion=True,
    help="List OpenRouter models with pricing",
    pretty_exceptions_enable=False,
)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


def fetch_openrouter_models(debug: bool = False) -> tuple[list[dict], dict[str, int]]:
    """Fetch models and pricing from OpenRouter API.

    Args:
        debug: Whether to print debug information

    Returns:
        Tuple of (models_list, stats) where:
        - models_list: List of model dicts with id, prompt, completion, avg, is_free
        - stats: Dict with 'total_models' and 'free_models' counts
    """
    print(f"Fetching models from {OPENROUTER_MODELS_URL}...")

    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error: API request failed: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    if "data" not in data:
        print("Error: Invalid API response (missing 'data' key)", file=sys.stderr)
        raise typer.Exit(code=1)

    models = []
    free_models_count = 0
    total_models_count = 0

    for model in data["data"]:
        model_id = model.get("id")
        pricing = model.get("pricing", {})

        if not model_id or not pricing:
            continue

        total_models_count += 1

        # Debug: log first model's structure
        if debug and len(models) == 0:
            print(f"\nDebug: Sample model structure for {model_id}:")
            print(f"  Raw pricing: {pricing}")

        # Convert string prices to floats and scale to per-1M tokens
        try:
            prompt_price = float(pricing.get("prompt", "0"))
            completion_price = float(pricing.get("completion", "0"))

            # Scale to per-1M tokens
            prompt_cost_per_1m = prompt_price * 1_000_000
            completion_cost_per_1m = completion_price * 1_000_000
            avg_cost_per_1m = (prompt_cost_per_1m + completion_cost_per_1m) / 2

            is_free = prompt_price == 0 and completion_price == 0
            if is_free:
                free_models_count += 1

            models.append({
                "id": model_id,
                "prompt": prompt_cost_per_1m,
                "completion": completion_cost_per_1m,
                "avg": avg_cost_per_1m,
                "is_free": is_free,
            })

        except (ValueError, TypeError) as e:
            if debug:
                print(f"Warning: Could not parse pricing for {model_id}: {e}", file=sys.stderr)
            continue

    stats = {
        "total_models": total_models_count,
        "free_models": free_models_count,
    }

    print(f"Fetched {len(models)} models")
    print(f"  - Total models: {total_models_count}")
    print(f"  - Free models: {free_models_count}\n")

    return models, stats


@app.command()
def list_models(
    free_only: Annotated[
        bool,
        typer.Option(
            "--free-only",
            help="Show only free models",
        ),
    ] = False,
    paid_only: Annotated[
        bool,
        typer.Option(
            "--paid-only",
            help="Show only paid models (non-free)",
        ),
    ] = False,
    sort_by: Annotated[
        str,
        typer.Option(
            "--sort-by",
            help="Sort by: 'avg' (average cost), 'prompt', 'completion', 'name' (default: avg)",
        ),
    ] = "avg",
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            "-n",
            help="Limit number of results shown",
        ),
    ] = None,
    reverse: Annotated[
        bool,
        typer.Option(
            "--reverse",
            "-r",
            help="Reverse sort order (most expensive first)",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Show debug information",
        ),
    ] = False,
) -> None:
    """List models from OpenRouter with pricing information.

    By default shows all models sorted by average cost (lowest first).
    Use filters to narrow down results.

    Examples:

        # List all paid models sorted by average cost (cheapest first)
        python scripts/list_openrouter_models.py --paid-only

        # List top 10 cheapest paid models
        python scripts/list_openrouter_models.py --paid-only --limit 10

        # List top 10 most expensive models
        python scripts/list_openrouter_models.py --paid-only --limit 10 --reverse

        # List all free models
        python scripts/list_openrouter_models.py --free-only

        # Sort by completion cost
        python scripts/list_openrouter_models.py --paid-only --sort-by completion

        # Show all models sorted by name
        python scripts/list_openrouter_models.py --sort-by name
    """
    print(f"OpenRouter Models Pricing")
    print(f"=========================\n")

    # Fetch models from OpenRouter
    models, api_stats = fetch_openrouter_models(debug)

    # Filter models
    models_to_show = []
    for model in models:
        # Apply filters
        if free_only and not model["is_free"]:
            continue
        if paid_only and model["is_free"]:
            continue

        models_to_show.append(model)

    # Sort models
    if sort_by == "name":
        models_to_show.sort(key=lambda x: x["id"], reverse=reverse)
    elif sort_by == "prompt":
        models_to_show.sort(key=lambda x: x["prompt"], reverse=reverse)
    elif sort_by == "completion":
        models_to_show.sort(key=lambda x: x["completion"], reverse=reverse)
    else:  # avg
        models_to_show.sort(key=lambda x: x["avg"], reverse=reverse)

    # Apply limit
    if limit:
        models_to_show = models_to_show[:limit]

    # Display results
    filter_desc = ""
    if free_only:
        filter_desc = " (free only)"
    elif paid_only:
        filter_desc = " (paid only)"

    sort_desc = f"{sort_by}"
    if reverse:
        sort_desc += " descending"
    else:
        sort_desc += " ascending"

    print(f"Showing {len(models_to_show)} models{filter_desc} (sorted by {sort_desc}):\n")
    print(f"{'Model ID':<60} {'Prompt/1M':<15} {'Completion/1M':<15} {'Avg/1M':<15}")
    print(f"{'-' * 60} {'-' * 15} {'-' * 15} {'-' * 15}")

    for model in models_to_show:
        if model["is_free"]:
            print(f"{model['id']:<60} {'FREE':<15} {'FREE':<15} {'FREE':<15}")
        else:
            prompt_str = f"${model['prompt']:.6f}"
            completion_str = f"${model['completion']:.6f}"
            avg_str = f"${model['avg']:.6f}"
            print(f"{model['id']:<60} {prompt_str:<15} {completion_str:<15} {avg_str:<15}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Displayed: {len(models_to_show)}")
    print(f"  Total available: {api_stats['total_models']}")
    print(f"  Free models: {api_stats['free_models']}")
    print(f"  Paid models: {api_stats['total_models'] - api_stats['free_models']}")


if __name__ == "__main__":
    app()
