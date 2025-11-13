"""Test that UUID generation natural keys align with ORM unique constraints.

This test ensures that:
1. Every ORM with UUID metadata has matching unique constraints
2. UUID functions exist and take the correct number of parameters
3. UUID function parameter counts match natural key length
4. UUID function parameter ORDER matches natural key ORDER
5. UUID function parameter NAMES roughly match column names

This prevents bugs where UUID generation doesn't match database uniqueness constraints,
which could lead to UUID collisions or constraint violations.
"""

import inspect
from sqlalchemy import UniqueConstraint

from llm_ensemble.infer.schemas import orms_normalized
from llm_ensemble.ingest.schemas import orms as ingest_orms
from llm_ensemble.libs.db import uuid_helpers


def get_all_orm_classes(module):
    """Extract all ORM classes from a module."""
    orms = []
    for name in dir(module):
        obj = getattr(module, name)
        # Check if it's a class and has __tablename__ (SQLAlchemy ORM marker)
        if (
            inspect.isclass(obj)
            and hasattr(obj, "__tablename__")
            and obj.__module__ == module.__name__
        ):
            orms.append(obj)
    return orms


def get_unique_constraint_columns(orm_class):
    """Extract columns from unique constraints for an ORM class.

    Returns a set of tuples, each representing a unique constraint's columns.
    Includes both single-column unique constraints and composite UniqueConstraint.
    """
    unique_constraints = set()

    # Check for single-column unique=True
    for column in orm_class.__table__.columns:
        if column.unique:
            unique_constraints.add((column.name,))

    # Check for composite UniqueConstraint in __table_args__
    if hasattr(orm_class, "__table_args__"):
        table_args = orm_class.__table_args__
        # __table_args__ can be a tuple or dict
        if isinstance(table_args, tuple):
            for arg in table_args:
                if isinstance(arg, UniqueConstraint):
                    # Extract column names from constraint
                    cols = tuple(col.name for col in arg.columns)
                    unique_constraints.add(cols)

    return unique_constraints


def normalize_name(name: str) -> str:
    """Normalize a name for comparison (lowercase, strip underscores)."""
    return name.lower().replace("_", "")


def test_uuid_alignment_for_all_orms():
    """Verify UUID generation natural keys match ORM unique constraints."""

    # Check both infer and ingest ORMs
    infer_orms = get_all_orm_classes(orms_normalized)
    ingest_orms_list = get_all_orm_classes(ingest_orms)
    all_orms = infer_orms + ingest_orms_list

    # Track ORMs we've validated
    validated_orms = []

    for orm_class in all_orms:
        # Skip if no UUID metadata (not all tables need deterministic UUIDs)
        if not hasattr(orm_class, "__uuid_function__"):
            continue

        # Require both metadata fields to be present
        assert hasattr(orm_class, "__natural_key__"), (
            f"{orm_class.__name__} has __uuid_function__ but missing __natural_key__"
        )

        natural_key = orm_class.__natural_key__
        uuid_func_name = orm_class.__uuid_function__

        # 1. Verify natural key matches a unique constraint
        unique_constraints = get_unique_constraint_columns(orm_class)
        assert natural_key in unique_constraints, (
            f"{orm_class.__name__}: Natural key {natural_key} doesn't match any "
            f"unique constraint. Available constraints: {unique_constraints}"
        )

        # 2. Verify UUID function exists
        assert hasattr(uuid_helpers, uuid_func_name), (
            f"{orm_class.__name__}: UUID function '{uuid_func_name}' not found in uuid_helpers"
        )

        uuid_func = getattr(uuid_helpers, uuid_func_name)

        # 3. Verify UUID function parameter count matches natural key length
        sig = inspect.signature(uuid_func)
        param_names = list(sig.parameters.keys())

        assert len(param_names) == len(natural_key), (
            f"{orm_class.__name__}: UUID function '{uuid_func_name}' takes "
            f"{len(param_names)} params {param_names} but natural key has {len(natural_key)} "
            f"columns {natural_key}"
        )

        # 4. Verify parameter ORDER and NAMES match natural key
        # We assume that if a uuid generating function takes the same input
        # as the natural key, then it will use these variables to compute the natural key
        for i, (param, col) in enumerate(zip(param_names, natural_key)):
            # Normalize both names for flexible comparison
            # This handles variations like "run_name" vs "runname" or case differences
            param_normalized = normalize_name(param)
            col_normalized = normalize_name(col)

            # Check if names match after normalization
            # This catches order issues since we're checking position i
            assert param_normalized == col_normalized, (
                f"{orm_class.__name__}: Parameter {i} is '{param}' but natural key column {i} "
                f"is '{col}'. Parameter order and names must match the natural key!\n"
                f"UUID function signature: {uuid_func_name}({', '.join(param_names)})\n"
                f"Natural key: {natural_key}"
            )

        validated_orms.append(orm_class.__name__)

    # Ensure we validated at least some ORMs (sanity check)
    # We expect at least 5 ingest + 8 infer = 13 total
    assert len(validated_orms) >= 10, (
        f"Expected to validate at least 10 ORMs, but only found {len(validated_orms)}: "
        f"{validated_orms}"
    )

    print(f"✓ Validated UUID alignment for {len(validated_orms)} ORMs: {validated_orms}")


def test_uuid_function_naming_convention():
    """Verify UUID functions follow naming conventions."""

    # Check both infer and ingest ORMs
    infer_orms = get_all_orm_classes(orms_normalized)
    ingest_orms_list = get_all_orm_classes(ingest_orms)
    all_orms = infer_orms + ingest_orms_list

    for orm_class in all_orms:
        if not hasattr(orm_class, "__uuid_function__"):
            continue

        uuid_func_name = orm_class.__uuid_function__

        # Should start with "compute_" and end with "_uuid"
        assert uuid_func_name.startswith("compute_"), (
            f"{orm_class.__name__}: UUID function '{uuid_func_name}' should start with 'compute_'"
        )
        assert uuid_func_name.endswith("_uuid"), (
            f"{orm_class.__name__}: UUID function '{uuid_func_name}' should end with '_uuid'"
        )


def test_natural_key_columns_exist():
    """Verify all columns in __natural_key__ actually exist in the ORM."""

    # Check both infer and ingest ORMs
    infer_orms = get_all_orm_classes(orms_normalized)
    ingest_orms_list = get_all_orm_classes(ingest_orms)
    all_orms = infer_orms + ingest_orms_list

    for orm_class in all_orms:
        if not hasattr(orm_class, "__natural_key__"):
            continue

        natural_key = orm_class.__natural_key__
        table_columns = {col.name for col in orm_class.__table__.columns}

        for col_name in natural_key:
            assert col_name in table_columns, (
                f"{orm_class.__name__}: Natural key references column '{col_name}' "
                f"which doesn't exist. Available columns: {table_columns}"
            )


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    test_uuid_alignment_for_all_orms()
    test_uuid_function_naming_convention()
    test_natural_key_columns_exist()
    print("✓ All UUID alignment tests passed!")
