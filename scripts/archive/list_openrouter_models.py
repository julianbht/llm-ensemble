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
        model_name = model.get("name", "")
        context_length = model.get("context_length")

        if not model_id or not pricing:
            continue

        total_models_count += 1

        # Debug: log first few models' structure
        if debug and len(models) < 3:
            print(f"\nDebug: Model structure for {model_id}:")
            print(f"  Model name: {model_name}")
            print(f"  Raw pricing: {pricing}")
            print(f"  Context length: {context_length}")

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

            # Try to extract parameter count from model_id or name
            # Heuristic: Look for patterns like "7b", "70B", "1.5b", "405b" in model ID/name
            # Regex explained: (\d+(?:\.\d+)?)[bB]
            #   - \d+ : one or more digits
            #   - (?:\.\d+)? : optionally a decimal point followed by digits (non-capturing group)
            #   - [bB] : followed by 'b' or 'B'
            # Examples: "7b", "1.5B", "70b", "405B"
            import re
            param_match = re.search(r'(\d+(?:\.\d+)?)[bB]', model_id + " " + model_name)
            param_size = None
            if param_match:
                # Convert to billions (e.g., "7b" -> 7.0, "1.5b" -> 1.5)
                param_size = float(param_match.group(1))

                # Debug: show param extraction for first few models
                if debug and len(models) < 3:
                    print(f"  ✓ Extracted param size: {param_size}B (from pattern '{param_match.group(0)}')")
            else:
                if debug and len(models) < 3:
                    print(f"  ✗ Could not extract param size from: '{model_id}' or '{model_name}'")

            models.append({
                "id": model_id,
                "name": model_name,
                "prompt": prompt_cost_per_1m,
                "completion": completion_cost_per_1m,
                "avg": avg_cost_per_1m,
                "is_free": is_free,
                "context_length": context_length,
                "param_size": param_size,  # in billions, or None if not found
            })

        except (ValueError, TypeError) as e:
            if debug:
                print(f"Warning: Could not parse pricing for {model_id}: {e}", file=sys.stderr)
            continue

    # Count models with detected parameter size
    models_with_params = sum(1 for m in models if m["param_size"] is not None)
    models_without_params = len(models) - models_with_params

    stats = {
        "total_models": total_models_count,
        "free_models": free_models_count,
        "models_with_params": models_with_params,
        "models_without_params": models_without_params,
    }

    print(f"Fetched {len(models)} models")
    print(f"  - Total models: {total_models_count}")
    print(f"  - Free models: {free_models_count}")
    print(f"  - Parameter size detected: {models_with_params} ({100 * models_with_params / total_models_count:.1f}%)")
    print(f"  - Parameter size not detected: {models_without_params} ({100 * models_without_params / total_models_count:.1f}%)\n")

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
    max_params: Annotated[
        Optional[float],
        typer.Option(
            "--max-params",
            help="Maximum parameter size in billions (e.g., 7, 13, 70)",
        ),
    ] = None,
    min_params: Annotated[
        Optional[float],
        typer.Option(
            "--min-params",
            help="Minimum parameter size in billions (e.g., 7, 13, 70)",
        ),
    ] = None,
    sort_by: Annotated[
        str,
        typer.Option(
            "--sort-by",
            help="Sort by: 'avg' (average cost), 'prompt', 'completion', 'name', 'params' (default: avg)",
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
    output: Annotated[
        Optional[str],
        typer.Option(
            "--output",
            "-o",
            help="Output model IDs to file (one per line)",
        ),
    ] = None,
    append: Annotated[
        bool,
        typer.Option(
            "--append",
            "-a",
            help="Append to output file instead of overwriting",
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

        # List paid models with max 13B parameters
        python scripts/list_openrouter_models.py --paid-only --max-params 13

        # List models between 7B and 70B parameters
        python scripts/list_openrouter_models.py --min-params 7 --max-params 70

        # Sort by parameter size
        python scripts/list_openrouter_models.py --sort-by params

        # Sort by completion cost
        python scripts/list_openrouter_models.py --paid-only --sort-by completion

        # Show all models sorted by name
        python scripts/list_openrouter_models.py --sort-by name

        # Save free models ≤ 8B to file
        python scripts/list_openrouter_models.py --free-only --max-params 8 --output my_models.txt

        # Append more models to the same file
        python scripts/list_openrouter_models.py --paid-only --max-params 8 --limit 5 --output my_models.txt --append
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

        # Parameter size filters
        if model["param_size"] is not None:
            if max_params is not None and model["param_size"] > max_params:
                continue
            if min_params is not None and model["param_size"] < min_params:
                continue
        else:
            # If param_size is None and filters are specified, skip this model
            if max_params is not None or min_params is not None:
                continue

        models_to_show.append(model)

    # Sort models
    if sort_by == "name":
        models_to_show.sort(key=lambda x: x["id"], reverse=reverse)
    elif sort_by == "prompt":
        models_to_show.sort(key=lambda x: x["prompt"], reverse=reverse)
    elif sort_by == "completion":
        models_to_show.sort(key=lambda x: x["completion"], reverse=reverse)
    elif sort_by == "params":
        # Put models without param_size at the end
        models_to_show.sort(key=lambda x: (x["param_size"] is None, x["param_size"] or 0), reverse=reverse)
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
    print(f"{'Model ID':<55} {'Params':<10} {'Prompt/1M':<15} {'Completion/1M':<15} {'Avg/1M':<15}")
    print(f"{'-' * 55} {'-' * 10} {'-' * 15} {'-' * 15} {'-' * 15}")

    for model in models_to_show:
        # Format parameter size
        if model["param_size"] is not None:
            params_str = f"{model['param_size']:.1f}B"
        else:
            params_str = "?"

        if model["is_free"]:
            print(f"{model['id']:<55} {params_str:<10} {'FREE':<15} {'FREE':<15} {'FREE':<15}")
        else:
            prompt_str = f"${model['prompt']:.6f}"
            completion_str = f"${model['completion']:.6f}"
            avg_str = f"${model['avg']:.6f}"
            print(f"{model['id']:<55} {params_str:<10} {prompt_str:<15} {completion_str:<15} {avg_str:<15}")

    # Write to output file if requested
    if output:
        mode = "a" if append else "w"
        with open(output, mode, encoding="utf-8") as f:
            for model in models_to_show:
                f.write(f"{model['id']}\n")

        action = "Appended" if append else "Wrote"
        print(f"\n{action} {len(models_to_show)} model IDs to: {output}")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"Summary:")
    print(f"  Displayed: {len(models_to_show)}")
    print(f"  Total available: {api_stats['total_models']}")
    print(f"  Free models: {api_stats['free_models']}")
    print(f"  Paid models: {api_stats['total_models'] - api_stats['free_models']}")
    print(f"\nParameter Size Detection:")
    print(f"  Models with param size: {api_stats['models_with_params']} ({100 * api_stats['models_with_params'] / api_stats['total_models']:.1f}%)")
    print(f"  Models without param size: {api_stats['models_without_params']} ({100 * api_stats['models_without_params'] / api_stats['total_models']:.1f}%)")


if __name__ == "__main__":
    app()
