#!/usr/bin/env python3
"""Generate JSON schemas from Pydantic models and consolidate in libs/schemas/.

This script:
1. Finds all Pydantic schema models across the codebase
2. Generates JSON schemas from them using Pydantic's model_json_schema()
3. Writes them to src/llm_ensemble/libs/schemas/ with consistent naming
4. Creates a schema index file for documentation

Usage:
    python scripts/generate_schemas.py
    make schemas
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from llm_ensemble.ingest.schemas.judging_example import JudgingExample
from llm_ensemble.ingest.schemas.query import Query
from llm_ensemble.ingest.schemas.document import Document
from llm_ensemble.ingest.schemas.relevance import Relevance
from llm_ensemble.infer.schemas.model_judgement_schema import ModelJudgement
from llm_ensemble.infer.schemas.model_config_schema import ModelConfig
from llm_ensemble.infer.schemas.prompt_config_schema import PromptConfig
from llm_ensemble.infer.schemas.io_config_schema import IOConfig
from llm_ensemble.aggregate.schemas.ensemble_result import EnsembleResult
from llm_ensemble.evaluate.schemas.evaluation_metrics import EvaluationMetrics


# Schema definitions: (category, model_class, output_filename, description)
# Categories determine subfolder structure in libs/schemas/
SCHEMAS = [
    # Data contract schemas (pipeline boundaries)
    ("data_contracts", JudgingExample, "sample.schema.json", "Normalized query-document pair (ingest → infer)"),
    ("data_contracts", ModelJudgement, "judgement.schema.json", "LLM judge output (infer → aggregate)"),
    ("data_contracts", EnsembleResult, "ensemble.schema.json", "Ensemble aggregation output (aggregate → evaluate)"),
    ("data_contracts", EvaluationMetrics, "metrics.schema.json", "Evaluation metrics output (evaluate → final)"),

    # Configuration schemas (YAML configs)
    ("configurations", ModelConfig, "model-config.schema.json", "Model configuration (configs/models/*.yaml)"),
    ("configurations", PromptConfig, "prompt-config.schema.json", "Prompt configuration (configs/prompts/*.yaml)"),
    ("configurations", IOConfig, "io-config.schema.json", "I/O format configuration (configs/io/*.yaml)"),

    # Internal domain models (CLI-specific, for documentation)
    ("internal", Query, "query.schema.json", "Query component (ingest-internal)"),
    ("internal", Document, "document.schema.json", "Document component (ingest-internal)"),
    ("internal", Relevance, "relevance.schema.json", "Relevance label component (ingest-internal)"),
]


def generate_json_schema(model_class: Any) -> dict:
    """Generate JSON schema from Pydantic model."""
    return model_class.model_json_schema()


def write_schema(schema: dict, output_path: Path, description: str) -> None:
    """Write schema to file with metadata."""
    # Add description to schema
    schema["description"] = description

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pretty-print JSON with 2-space indentation
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
        f.write("\n")  # Add trailing newline

    print(f"✓ Generated {output_path.relative_to(output_path.parent.parent)}")


def generate_schema_index(schemas_dir: Path, schemas_by_category: dict) -> None:
    """Generate an index file listing all schemas organized by category."""
    categories = {
        "data_contracts": {
            "description": "Schemas defining data flow between CLI stages",
            "path": "data_contracts/",
            "schemas": schemas_by_category.get("data_contracts", []),
            "pipeline": "ingest → infer → aggregate → evaluate"
        },
        "configurations": {
            "description": "Schemas for YAML configuration files",
            "path": "configurations/",
            "schemas": schemas_by_category.get("configurations", [])
        },
        "internal": {
            "description": "Internal domain models for documentation (CLI-specific, not pipeline contracts)",
            "path": "internal/",
            "schemas": schemas_by_category.get("internal", [])
        }
    }

    index = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LLM Ensemble Schema Index",
        "description": "Index of all schemas in the llm-ensemble project. All schemas are auto-generated from Pydantic models. Schemas are organized into subfolders by category.",
        "categories": categories
    }

    index_path = schemas_dir / "schema-index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"✓ Generated schema-index.json")


def main() -> None:
    """Main entry point."""
    # Determine schemas directory
    schemas_dir = project_root / "src" / "llm_ensemble" / "libs" / "schemas"

    print(f"Generating schemas in {schemas_dir.relative_to(project_root)}\n")

    # Track schemas by category for index generation
    schemas_by_category = {}

    # Generate schemas
    for category, model_class, filename, description in SCHEMAS:
        # Create category subdirectory
        category_dir = schemas_dir / category
        output_path = category_dir / filename

        try:
            schema = generate_json_schema(model_class)
            write_schema(schema, output_path, description)

            # Track for index
            if category not in schemas_by_category:
                schemas_by_category[category] = []
            schemas_by_category[category].append(filename)

        except Exception as e:
            print(f"✗ Failed to generate {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    # Generate index
    print()
    generate_schema_index(schemas_dir, schemas_by_category)

    print(f"\n✓ Successfully generated {len(SCHEMAS)} schemas + index")
    print(f"\nSchemas are now organized in:")
    for category in sorted(schemas_by_category.keys()):
        print(f"  {schemas_dir.relative_to(project_root)}/{category}/")


if __name__ == "__main__":
    main()
