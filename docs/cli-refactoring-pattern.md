# CLI Refactoring Pattern

## Overview

The infer CLI has been refactored to separate the entry point from orchestration logic, following clean architecture principles. This pattern should be applied to other CLIs (aggregate, evaluate) as they are implemented.

## Architecture

### Before Refactoring

```
infer_cli.py (196 lines)
├── Typer app setup
├── Helper functions (_read_examples, _json_dumps)
├── @app.command decorator
└── infer() function with ALL logic:
    ├── Config loading
    ├── Run directory setup
    ├── Logger initialization
    ├── File I/O
    ├── Inference orchestration
    ├── Progress tracking
    ├── Manifest writing
    └── Cleanup
```

### After Refactoring

```
infer_cli.py (84 lines) - THIN ENTRY POINT
├── Typer app setup
├── @app.command decorator
└── infer() function - DELEGATES to orchestrator
    └── Error handling only

infer/orchestrator.py (NEW, 212 lines) - ORCHESTRATION LOGIC
├── Helper functions (_read_examples, _json_dumps)
└── run_inference() function with ALL business logic:
    ├── Config loading
    ├── Run directory setup
    ├── Logger initialization
    ├── File I/O
    ├── Inference orchestration
    ├── Progress tracking
    ├── Manifest writing
    └── Cleanup
```

## Benefits

1. **Testability**: Orchestrator can be tested without CLI framework
2. **Reusability**: Orchestration logic can be imported by other modules
3. **Separation of Concerns**: CLI handles arg parsing, orchestrator handles business logic
4. **Consistency**: All CLIs follow the same pattern (ingest also follows this)
5. **Maintainability**: Each module has a single, clear responsibility

## Pattern Structure

### CLI Entry Point (`<name>_cli.py`)

**Responsibilities:**
- Define CLI interface with Typer
- Parse and validate command-line arguments
- Delegate to orchestrator
- Handle top-level exceptions and exit codes

**Template:**
```python
from llm_ensemble.<module>.orchestrator import run_<operation>
from llm_ensemble.libs.runtime.env import load_runtime_config

load_runtime_config()

app = typer.Typer(add_completion=False, help="Description")

@app.command("<command>")
def <command>(...args):
    """Command docstring."""
    try:
        run_<operation>(...args)
    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
```

### Orchestrator (`<module>/orchestrator.py`)

**Responsibilities:**
- Load configurations
- Set up run directory and artifacts
- Initialize logger
- Execute core business logic
- Write outputs and manifests
- Return metadata about the run

**Template:**
```python
"""Orchestrator for the <module> CLI.

This module contains the top-level orchestration logic for <operation>.
It is separated from the CLI entry point to enable better testability and reusability.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, TextIO

# Import domain and adapter modules
from llm_ensemble.<module>.domain import ...
from llm_ensemble.<module>.adapters import ...
from llm_ensemble.libs.runtime.run_manager import create_run_id, get_run_dir, write_manifest
from llm_ensemble.libs.logging.logger import get_logger


def run_<operation>(
    # All CLI parameters
    ...
    log_file: Optional[TextIO] = None,  # For testing
) -> dict:
    """Run <operation> with full orchestration.
    
    Args:
        ...: CLI parameters
        log_file: Optional file handle for logging (used when save_logs=True)
        
    Returns:
        Dictionary with run metadata including:
        - run_id: The run identifier
        - run_dir: Path to run directory
        - output_file: Path to output file
        - ... other relevant metrics
        
    Raises:
        FileNotFoundError: If config not found
        Exception: If operation fails
    """
    # 1. Load configuration
    # 2. Set up run directory
    # 3. Initialize logger
    # 4. Execute core logic
    # 5. Write manifest
    # 6. Cleanup
    # 7. Return metadata
    
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "output_file": output_file,
        # ... other metadata
    }
```

## Ingest CLI Structure

The ingest CLI now follows the same refactored pattern:

```
ingest_cli.py (69 lines) - THIN ENTRY POINT
├── Typer CLI setup
├── Argument definitions
└── Delegates to orchestrator

ingest/orchestrator.py (157 lines) - ORCHESTRATION LOGIC
├── Helper functions (_json_dumps)
└── run_ingest() function with ALL business logic:
    ├── Config loading
    ├── Adapter loading
    ├── Run directory setup
    ├── Logger initialization
    ├── Example processing
    ├── Manifest writing
    └── Cleanup
```

## Testing Strategy

### Before Refactoring
- Test CLI via subprocess or Typer CliRunner
- Difficult to mock file I/O and logging
- Hard to test individual components

### After Refactoring
- **Unit tests**: Test orchestrator functions directly
- **Integration tests**: Test orchestrator with real files
- **CLI tests**: Minimal tests for CLI entry point
- **Mocking**: Easy to inject mock dependencies into orchestrator

Example orchestrator test:
```python
from llm_ensemble.infer.orchestrator import run_inference

def test_run_inference(tmp_path):
    input_file = tmp_path / "input.ndjson"
    input_file.write_text('{"query": "test", ...}\n')
    
    result = run_inference(
        model="test-model",
        input_file=input_file,
        limit=1,
        save_logs=False,
    )
    
    assert result["judgement_count"] == 1
    assert result["output_file"].exists()
```

## Next Steps

1. ✅ **infer CLI**: DONE - Refactored with orchestrator pattern
2. ✅ **ingest CLI**: DONE - Refactored with orchestrator pattern  
3. **aggregate CLI**: When implementing, use this pattern from the start
4. **evaluate CLI**: When implementing, use this pattern from the start

## File Organization

```
src/llm_ensemble/
├── infer_cli.py          # ✅ Thin CLI entry point
├── ingest_cli.py         # ✅ Thin CLI entry point
├── aggregate_cli.py      # Future: thin CLI entry point
├── evaluate_cli.py       # Future: thin CLI entry point
├── infer/
│   ├── orchestrator.py   # ✅ Orchestration logic
│   ├── domain/           # Pure business logic
│   ├── adapters/         # I/O and external services
│   └── ...
├── ingest/
│   ├── orchestrator.py   # ✅ Orchestration logic
│   ├── domain/
│   ├── adapters/
│   └── ...
├── aggregate/
│   ├── orchestrator.py   # FUTURE: Implement with this pattern
│   └── ...
└── evaluate/
    ├── orchestrator.py   # FUTURE: Implement with this pattern
    └── ...
```

## Key Principles

1. **CLI is a thin wrapper** - Only argument parsing and error handling
2. **Orchestrator is the engine** - Contains all business logic
3. **Return metadata** - Orchestrator returns structured data about what was done
4. **Testability first** - Design for easy unit and integration testing
5. **Consistency** - All CLIs follow the same pattern
6. **Separation of concerns** - Each module has one clear responsibility

## Migration Guide for Other CLIs

When implementing aggregate/evaluate CLIs:

1. Create `<module>/orchestrator.py` first
2. Define `run_<operation>()` function with all logic
3. Create thin CLI wrapper that delegates to orchestrator
4. Write tests for orchestrator (easy!)
5. Write minimal CLI tests (just entry point validation)

This approach ensures clean, testable, maintainable code from day one.
