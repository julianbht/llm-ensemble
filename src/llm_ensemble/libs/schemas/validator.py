"""JSON Schema validation utilities for NDJSON output files.

Validates CLI outputs against their canonical schemas.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, Optional

try:
    from jsonschema import validate, ValidationError, Draft202012Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    ValidationError = Exception  # Fallback type


def get_schema_path(schema_name: str) -> Path:
    """Get the path to a JSON schema file.

    Args:
        schema_name: Name of the schema (e.g., 'sample', 'judgement', 'ensemble', 'metrics')

    Returns:
        Path to the schema file

    Example:
        >>> get_schema_path('sample')
        PosixPath('.../libs/schemas/sample.schema.json')
    """
    schema_dir = Path(__file__).parent
    return schema_dir / f"{schema_name}.schema.json"


def load_schema(schema_name: str) -> dict:
    """Load a JSON schema from file.

    Args:
        schema_name: Name of the schema (e.g., 'sample', 'judgement')

    Returns:
        Schema dictionary

    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema is invalid JSON

    Example:
        >>> schema = load_schema('sample')
        >>> schema['title']
        'JudgingExample'
    """
    schema_path = get_schema_path(schema_name)
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_record(record: dict, schema_name: str) -> tuple[bool, Optional[str]]:
    """Validate a single record against a schema.

    Args:
        record: Record to validate (as dict)
        schema_name: Name of the schema to validate against

    Returns:
        Tuple of (is_valid, error_message)
        - (True, None) if valid
        - (False, error_message) if invalid

    Example:
        >>> record = {"dataset": "llm-judge-2024", "query_id": "q1", ...}
        >>> is_valid, error = validate_record(record, "sample")
        >>> assert is_valid
    """
    if not JSONSCHEMA_AVAILABLE:
        return False, "jsonschema library not installed. Run: pip install jsonschema"

    try:
        schema = load_schema(schema_name)
        Draft202012Validator(schema).validate(record)
        return True, None
    except ValidationError as e:
        return False, f"Validation error: {e.message} at {'.'.join(str(p) for p in e.path)}"
    except FileNotFoundError as e:
        return False, f"Schema not found: {e}"
    except json.JSONDecodeError as e:
        return False, f"Invalid schema JSON: {e}"


def validate_ndjson_file(
    file_path: Path,
    schema_name: str,
    max_errors: int = 10
) -> tuple[int, int, list[str]]:
    """Validate an NDJSON file against a schema.

    Args:
        file_path: Path to NDJSON file
        schema_name: Name of schema to validate against
        max_errors: Maximum number of errors to collect (default 10)

    Returns:
        Tuple of (valid_count, invalid_count, error_messages)

    Example:
        >>> valid, invalid, errors = validate_ndjson_file(
        ...     Path("samples.ndjson"), "sample"
        ... )
        >>> print(f"Valid: {valid}, Invalid: {invalid}")
        Valid: 100, Invalid: 0
    """
    if not JSONSCHEMA_AVAILABLE:
        return 0, 0, ["jsonschema library not installed. Run: pip install jsonschema"]

    valid_count = 0
    invalid_count = 0
    errors = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                is_valid, error = validate_record(record, schema_name)

                if is_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
                    if len(errors) < max_errors:
                        errors.append(f"Line {line_num}: {error}")

            except json.JSONDecodeError as e:
                invalid_count += 1
                if len(errors) < max_errors:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")

    if invalid_count > max_errors:
        errors.append(f"... and {invalid_count - max_errors} more errors")

    return valid_count, invalid_count, errors


def iter_validated_records(
    file_path: Path,
    schema_name: str,
    skip_invalid: bool = False
) -> Iterator[tuple[dict, bool, Optional[str]]]:
    """Iterate over records in an NDJSON file with validation.

    Args:
        file_path: Path to NDJSON file
        schema_name: Name of schema to validate against
        skip_invalid: If True, skip invalid records; if False, yield them anyway

    Yields:
        Tuples of (record, is_valid, error_message)

    Example:
        >>> for record, is_valid, error in iter_validated_records(Path("samples.ndjson"), "sample"):
        ...     if not is_valid:
        ...         print(f"Invalid record: {error}")
        ...     else:
        ...         # Process valid record
        ...         pass
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                is_valid, error = validate_record(record, schema_name)

                if skip_invalid and not is_valid:
                    continue

                yield record, is_valid, error

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON: {e}"
                if not skip_invalid:
                    yield {}, False, error_msg
