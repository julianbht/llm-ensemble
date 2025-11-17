# UX Improvements Implementation Plan

## Overview

Three user experience improvements to implement across all four CLIs (ingest, infer, aggregate, evaluate):

1. **Display User Misses** - Show available configs when required flags are missing
2. **Run References / Shortcuts** - Tag and reference runs by aliases instead of full paths
3. **Run Templates / Profiles** - Save and reuse frequently-used configurations

**Recommended implementation order:** 1 → 2 → 3 (increasing complexity, immediate value first)

---

## Feature 1: Display User Misses ⭐ START HERE

### Problem
When users forget required flags, they get generic Typer errors without context about available options.

**Current behavior:**
```bash
$ infer --io json
Error: Missing option '--model-cfg'
```

**Desired behavior:**
```bash
$ infer --io json
Error: Missing required option '--model-cfg'

Available model configs in configs/models/:
  - gemma-3n-e2b-it-free
  - gpt-oss-20b-free
  - llama-4-maverick-free
  - mai-ds-r1:free
  - nemotron-nano-9b-v2-free
  - openai-gpt-5.1

Example:
  infer --model-cfg gpt-oss-20b-free --prompt-cfg thomas-simple --io-cfg json
```

### Benefits
- **Immediate value** - Helps users discover available configs without reading docs
- **Low complexity** - No state management, just better error messages
- **Educational** - Teaches users the config-first design pattern

### Implementation Strategy

#### A. Custom Typer Callback Validation (Recommended)

Use Typer's callback mechanism to validate and provide rich error messages before command execution.

**Why this approach:**
- ✅ Catches missing flags before command logic runs
- ✅ Can provide context-aware suggestions based on CLI
- ✅ Clean separation - validation logic separate from business logic
- ✅ Consistent across all CLIs via shared helper functions

**Architecture:**

```
libs/cli/
├── common_params.py          # Existing - parameter definitions
├── validation_callbacks.py   # NEW - Validation functions for required params
└── error_messages.py          # NEW - Rich error message formatters
```

#### B. Implementation Steps

**Step 1: Create validation helpers**

```python
# libs/cli/validation_callbacks.py
"""Validation callbacks for CLI parameters with rich error messages."""

from pathlib import Path
from typing import Optional
import typer

from llm_ensemble.libs.runtime.path_manager import PathManager
from llm_ensemble.libs.cli.error_messages import format_missing_config_error


def validate_model_config(value: Optional[str]) -> Optional[str]:
    """Validate model config and show available options if missing."""
    if value is None:
        available = list_available_configs(PathManager.get_model_configs_dir())
        raise typer.BadParameter(
            format_missing_config_error(
                param_name="--model-cfg",
                config_type="model",
                config_dir=PathManager.get_model_configs_dir(),
                available=available,
                example="gpt-oss-20b-free"
            )
        )
    return value


def validate_prompt_config(value: Optional[str]) -> Optional[str]:
    """Validate prompt config and show available options if missing."""
    if value is None:
        available = list_available_configs(PathManager.get_prompts_dir())
        raise typer.BadParameter(
            format_missing_config_error(
                param_name="--prompt-cfg",
                config_type="prompt",
                config_dir=PathManager.get_prompts_dir(),
                available=available,
                example="thomas-simple"
            )
        )
    return value


def list_available_configs(config_dir: Path) -> list[str]:
    """List available YAML configs in directory."""
    if not config_dir.exists():
        return []
    return sorted([p.stem for p in config_dir.glob("*.yaml")])
```

```python
# libs/cli/error_messages.py
"""Rich error message formatters for CLI validation errors."""

from pathlib import Path
from typing import List


def format_missing_config_error(
    param_name: str,
    config_type: str,
    config_dir: Path,
    available: List[str],
    example: str,
) -> str:
    """Format a rich error message for missing config parameter.
    
    Args:
        param_name: CLI flag name (e.g., "--model-cfg")
        config_type: Human-readable config type (e.g., "model", "prompt")
        config_dir: Directory containing configs
        available: List of available config names
        example: Example config name to show in usage
    
    Returns:
        Formatted error message with available options and example
    """
    relative_dir = config_dir.relative_to(Path.cwd().parent if "llm-ensemble" in str(Path.cwd()) else Path.cwd())
    
    lines = [
        f"\nMissing required option: {param_name}",
        f"\nAvailable {config_type} configs in {relative_dir}/:",
    ]
    
    if available:
        lines.extend([f"  • {name}" for name in available])
    else:
        lines.append(f"  (No {config_type} configs found)")
    
    lines.extend([
        f"\nExample usage:",
        f"  <command> {param_name} {example} ...",
    ])
    
    return "\n".join(lines)


def format_missing_io_config_error(
    cli_name: str,
    config_dir: Path,
    available: List[str],
) -> str:
    """Format error message for missing I/O config (CLI-specific)."""
    return format_missing_config_error(
        param_name="--io-cfg",
        config_type=f"{cli_name} I/O",
        config_dir=config_dir,
        available=available,
        example=available[0] if available else "json",
    )
```

**Step 2: Update common_params.py to use callbacks**

```python
# libs/cli/common_params.py (additions)
from llm_ensemble.libs.cli.validation_callbacks import (
    validate_model_config,
    validate_prompt_config,
)

# Add to infer-specific params
ModelCfg = Annotated[
    str,
    typer.Option(
        ...,  # Required
        "--model-cfg",
        callback=validate_model_config,
        help=f"Model config name. Configs in {PathManager.get_model_configs_dir().relative_to(PathManager.get_project_root())}"
    )
]

PromptCfg = Annotated[
    str,
    typer.Option(
        ...,  # Required
        "--prompt-cfg",
        callback=validate_prompt_config,
        help=f"Prompt config name. Configs in {PathManager.get_prompts_dir().relative_to(PathManager.get_project_root())}"
    )
]
```

**Step 3: Update CLIs to use new typed parameters**

```python
# infer_cli.py (change)
from llm_ensemble.libs.cli.common_params import ModelCfg, PromptCfg

@app.command("infer")
def infer(
    io_cfg: IoCfg,
    model_cfg: ModelCfg,  # Now has validation callback
    prompt_cfg: PromptCfg,  # Now has validation callback
    # ... rest of params
):
    # ... existing implementation
```

**Step 4: Add I/O-specific validation**

I/O configs are CLI-specific, so we need a factory function:

```python
# libs/cli/validation_callbacks.py (addition)

def make_io_config_validator(cli_name: str):
    """Create an I/O config validator for specific CLI."""
    def validate_io_config(value: Optional[str]) -> Optional[str]:
        if value is None:
            config_dir = PathManager.get_io_configs_dir(cli_name)
            available = list_available_configs(config_dir)
            raise typer.BadParameter(
                format_missing_io_config_error(cli_name, config_dir, available)
            )
        return value
    return validate_io_config
```

```python
# infer_cli.py (change at top)
from llm_ensemble.libs.cli.validation_callbacks import make_io_config_validator

# Create CLI-specific I/O validator
validate_infer_io_config = make_io_config_validator("infer")

# Use in parameter definition
io_cfg: Annotated[str, typer.Option(
    ...,
    "--io-cfg",
    callback=validate_infer_io_config,
    help="..."
)]
```

#### C. Testing Strategy

```python
# tests/libs/cli/test_validation_callbacks.py
"""Tests for CLI validation callbacks."""

def test_validate_model_config_missing():
    """Should raise with available configs when None."""
    with pytest.raises(typer.BadParameter) as exc_info:
        validate_model_config(None)
    
    error_msg = str(exc_info.value)
    assert "--model-cfg" in error_msg
    assert "Available model configs" in error_msg
    assert "gpt-oss-20b-free" in error_msg  # Should list actual configs


def test_validate_model_config_valid():
    """Should return value when provided."""
    result = validate_model_config("gpt-oss-20b-free")
    assert result == "gpt-oss-20b-free"
```

#### D. Impact on All CLIs

| CLI       | Required Flags with Validation                  |
|-----------|-------------------------------------------------|
| ingest    | `--io-cfg` (CLI-specific validator)             |
| infer     | `--model-cfg`, `--prompt-cfg`, `--io-cfg`       |
| aggregate | `--ensemble-cfg`, `--io-cfg`                    |
| evaluate  | `--io-cfg` (when fully implemented)             |

#### E. Effort Estimate

- **Files to create:** 2 (validation_callbacks.py, error_messages.py)
- **Files to modify:** 5 (common_params.py, 4 CLI files)
- **Tests to write:** 3-4 test files
- **Time estimate:** 4-6 hours
- **Risk:** Low (pure validation layer, doesn't touch business logic)

---

## Feature 2: Run References / Shortcuts

### Problem
Users must reference runs by full paths, which are long and hard to remember:
```bash
aggregate --input artifacts/runs/infer/test/20250128_143022_gpt-oss-20b/judgements.json
```

**Desired UX:**
```bash
# Tag a run after completion
infer ... --tag baseline

# Or tag existing run
llm-ensemble tag add infer/test/20250128_143022_gpt-oss-20b baseline

# Reference by tag
aggregate --input @baseline --ensemble weighted_majority

# List tags
llm-ensemble tag list
llm-ensemble tag list --cli infer

# Remove tag
llm-ensemble tag remove baseline
```

### Benefits
- **Workflow improvement** - Easy to reference runs in multi-stage pipelines
- **Reproducibility** - Named references make experiments more memorable
- **Shareability** - Team members can reference canonical runs by name

### Implementation Strategy

#### A. Tag Storage Design

**Option 1: File-based tags (Recommended)**
```
artifacts/runs/.tags/
├── baseline -> infer/test/20250128_143022_gpt-oss-20b/
├── prod-v1 -> aggregate/official/20250129_091234_weighted/
└── experiment-1 -> infer/test/20250128_150000_llama-4/
```

**Why symlinks:**
- ✅ Native filesystem feature (works on Linux/macOS/Windows)
- ✅ `ls -la` shows targets naturally
- ✅ No parsing needed - just `readlink()`
- ✅ Git-friendly (can commit official run tags)

**Option 2: JSON registry** (Alternative)
```json
{
  "baseline": {
    "run_path": "infer/test/20250128_143022_gpt-oss-20b",
    "created_at": "2025-01-28T14:30:22Z",
    "created_by": "user",
    "notes": "Initial baseline"
  }
}
```

#### B. Architecture

```
libs/runtime/
├── run_tag_manager.py   # NEW - Tag CRUD operations
└── tag_resolver.py      # NEW - Resolve @tag to Path

libs/cli/
└── tag_cli.py           # NEW - llm-ensemble tag subcommands
```

#### C. Core Implementation

```python
# libs/runtime/run_tag_manager.py
"""Manage run tags using filesystem symlinks."""

from pathlib import Path
from typing import List, Optional, Dict
import os

from llm_ensemble.libs.runtime.path_manager import PathManager


class RunTagManager:
    """Manages tags for referencing runs by short aliases.
    
    Tags are stored as symlinks in artifacts/runs/.tags/
    pointing to relative run directories (e.g., infer/test/run_name/).
    """
    
    @staticmethod
    def get_tags_dir() -> Path:
        """Get the tags directory."""
        return PathManager.get_artifacts_dir() / "runs" / ".tags"
    
    @staticmethod
    def create_tag(tag_name: str, run_path: Path, overwrite: bool = False) -> None:
        """Create a tag pointing to a run directory.
        
        Args:
            tag_name: Short alias for the run (e.g., "baseline")
            run_path: Path to run directory (absolute or relative to artifacts/runs/)
            overwrite: If True, replace existing tag
        
        Raises:
            ValueError: If tag already exists and overwrite=False
            FileNotFoundError: If run_path doesn't exist
        """
        tags_dir = RunTagManager.get_tags_dir()
        tags_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate tag name (alphanumeric + hyphens/underscores)
        if not tag_name.replace("-", "").replace("_", "").isalnum():
            raise ValueError(f"Invalid tag name: {tag_name} (use alphanumeric, hyphens, underscores)")
        
        # Make run_path relative to artifacts/runs/ for symlink
        runs_dir = PathManager.get_artifacts_dir() / "runs"
        if run_path.is_absolute():
            try:
                relative_run_path = run_path.relative_to(runs_dir)
            except ValueError:
                raise ValueError(f"Run path must be under {runs_dir}")
        else:
            relative_run_path = run_path
        
        # Check if run exists
        full_run_path = runs_dir / relative_run_path
        if not full_run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {full_run_path}")
        
        # Check if tag already exists
        tag_path = tags_dir / tag_name
        if tag_path.exists() and not overwrite:
            existing_target = os.readlink(tag_path)
            raise ValueError(
                f"Tag '{tag_name}' already exists (points to {existing_target}). "
                f"Use --overwrite to replace."
            )
        
        # Create symlink (relative to parent directory for portability)
        # Target: ../infer/test/run_name (from .tags/ to runs/cli/type/name/)
        relative_target = Path("..") / relative_run_path
        
        if tag_path.exists():
            tag_path.unlink()
        
        tag_path.symlink_to(relative_target)
    
    @staticmethod
    def get_tag(tag_name: str) -> Optional[Path]:
        """Resolve a tag to its run directory path.
        
        Args:
            tag_name: Tag to resolve
        
        Returns:
            Absolute path to run directory, or None if tag doesn't exist
        """
        tag_path = RunTagManager.get_tags_dir() / tag_name
        if not tag_path.exists():
            return None
        
        # Read symlink and resolve to absolute path
        target = os.readlink(tag_path)
        runs_dir = PathManager.get_artifacts_dir() / "runs"
        return (RunTagManager.get_tags_dir() / target).resolve()
    
    @staticmethod
    def list_tags(cli_filter: Optional[str] = None) -> Dict[str, Path]:
        """List all tags, optionally filtered by CLI.
        
        Args:
            cli_filter: Only show tags for this CLI (e.g., "infer")
        
        Returns:
            Dict mapping tag names to absolute run paths
        """
        tags_dir = RunTagManager.get_tags_dir()
        if not tags_dir.exists():
            return {}
        
        tags = {}
        for tag_path in tags_dir.iterdir():
            if not tag_path.is_symlink():
                continue
            
            tag_name = tag_path.name
            run_path = RunTagManager.get_tag(tag_name)
            
            if run_path is None:
                continue  # Broken symlink
            
            # Apply CLI filter
            if cli_filter:
                # Extract CLI name from path: runs/infer/test/name -> "infer"
                try:
                    parts = run_path.relative_to(PathManager.get_artifacts_dir() / "runs").parts
                    if parts[0] != cli_filter:
                        continue
                except ValueError:
                    continue  # Path not under runs/
            
            tags[tag_name] = run_path
        
        return tags
    
    @staticmethod
    def remove_tag(tag_name: str) -> bool:
        """Remove a tag.
        
        Args:
            tag_name: Tag to remove
        
        Returns:
            True if tag was removed, False if it didn't exist
        """
        tag_path = RunTagManager.get_tags_dir() / tag_name
        if not tag_path.exists():
            return False
        
        tag_path.unlink()
        return True
```

```python
# libs/runtime/tag_resolver.py
"""Resolve @tag references in CLI inputs."""

from pathlib import Path
from typing import Union

from llm_ensemble.libs.runtime.run_tag_manager import RunTagManager


def resolve_input_path(path_or_tag: Union[str, Path]) -> Path:
    """Resolve a path or @tag reference to an absolute path.
    
    Args:
        path_or_tag: Regular path or tag reference (e.g., "@baseline")
    
    Returns:
        Resolved absolute path
    
    Raises:
        ValueError: If tag doesn't exist
    """
    path_str = str(path_or_tag)
    
    # Check if it's a tag reference
    if path_str.startswith("@"):
        tag_name = path_str[1:]  # Remove @
        run_path = RunTagManager.get_tag(tag_name)
        
        if run_path is None:
            # List available tags for helpful error
            available_tags = list(RunTagManager.list_tags().keys())
            tags_str = ", ".join(available_tags) if available_tags else "(none)"
            raise ValueError(
                f"Tag not found: @{tag_name}\n"
                f"Available tags: {tags_str}\n"
                f"Create tags with: llm-ensemble tag add <run-path> {tag_name}"
            )
        
        return run_path
    
    # Regular path - convert to absolute
    path = Path(path_str)
    if not path.is_absolute():
        path = Path.cwd() / path
    
    return path
```

#### D. CLI Integration

**1. Standalone tag management CLI:**

```python
# src/llm_ensemble/tag_cli.py
"""Tag management CLI - llm-ensemble tag subcommands."""

import typer
from rich.console import Console
from rich.table import Table

from llm_ensemble.libs.runtime.run_tag_manager import RunTagManager
from llm_ensemble.libs.runtime.path_manager import PathManager

app = typer.Typer(help="Manage run tags")
console = Console()


@app.command("add")
def add_tag(
    run_path: str = typer.Argument(..., help="Run directory path (e.g., infer/test/run_name)"),
    tag_name: str = typer.Argument(..., help="Tag name (e.g., baseline)"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace existing tag"),
):
    """Create a tag for a run."""
    try:
        RunTagManager.create_tag(tag_name, Path(run_path), overwrite=overwrite)
        console.print(f"✓ Tagged [cyan]{run_path}[/cyan] as [green]@{tag_name}[/green]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_tags(
    cli: str = typer.Option(None, "--cli", help="Filter by CLI (infer, aggregate, etc.)"),
):
    """List all tags."""
    tags = RunTagManager.list_tags(cli_filter=cli)
    
    if not tags:
        console.print("No tags found.")
        return
    
    table = Table(title="Run Tags")
    table.add_column("Tag", style="green")
    table.add_column("Run Path", style="cyan")
    
    runs_dir = PathManager.get_artifacts_dir() / "runs"
    for tag_name, run_path in sorted(tags.items()):
        relative_path = run_path.relative_to(runs_dir) if run_path.is_relative_to(runs_dir) else run_path
        table.add_row(f"@{tag_name}", str(relative_path))
    
    console.print(table)


@app.command("remove")
def remove_tag(
    tag_name: str = typer.Argument(..., help="Tag to remove"),
):
    """Remove a tag."""
    if RunTagManager.remove_tag(tag_name):
        console.print(f"✓ Removed tag [green]@{tag_name}[/green]")
    else:
        console.print(f"[yellow]Tag not found:[/yellow] @{tag_name}")
        raise typer.Exit(1)


@app.command("get")
def get_tag(
    tag_name: str = typer.Argument(..., help="Tag to resolve"),
):
    """Resolve a tag to its run path."""
    run_path = RunTagManager.get_tag(tag_name)
    if run_path:
        console.print(str(run_path))
    else:
        console.print(f"[red]Tag not found:[/red] @{tag_name}")
        raise typer.Exit(1)
```

**2. Add --tag to existing CLIs:**

```python
# libs/cli/common_params.py (addition)
Tag = Annotated[
    Optional[str],
    typer.Option(
        "--tag",
        help="Create a tag for this run (e.g., --tag baseline)"
    )
]
```

```python
# infer_cli.py (additions)
from llm_ensemble.libs.cli.common_params import Tag
from llm_ensemble.libs.runtime.run_tag_manager import RunTagManager

@app.command("infer")
def infer(
    # ... existing params
    tag: Tag = None,
):
    """Run LLM inference..."""
    # ... existing logic ...
    
    # After successful run, create tag if requested
    if tag:
        try:
            # Get relative path from artifacts/runs/
            runs_dir = PathManager.get_artifacts_dir() / "runs"
            relative_path = run_dir.relative_to(runs_dir)
            RunTagManager.create_tag(tag, relative_path)
            logger.info(f"Tagged run as @{tag}")
        except Exception as e:
            logger.warning(f"Failed to create tag: {e}")
```

**3. Update input path handling to support @tags:**

```python
# infer_cli.py (change)
from llm_ensemble.libs.runtime.tag_resolver import resolve_input_path

@app.command("infer")
def infer(
    input_path: InputPath = None,
    # ...
):
    """Run LLM inference..."""
    
    # Resolve @tag references
    if input_path:
        input_path = resolve_input_path(input_path)
    
    # ... rest of logic
```

#### E. Testing Strategy

```python
# tests/libs/runtime/test_run_tag_manager.py
"""Tests for run tag management."""

def test_create_and_get_tag(tmp_path):
    """Should create tag and resolve to run path."""
    # Create fake run directory
    run_dir = tmp_path / "runs" / "infer" / "test" / "my-run"
    run_dir.mkdir(parents=True)
    
    # Create tag
    RunTagManager.create_tag("baseline", run_dir.relative_to(tmp_path / "runs"))
    
    # Resolve tag
    resolved = RunTagManager.get_tag("baseline")
    assert resolved == run_dir


def test_resolve_input_path_with_tag():
    """Should resolve @tag to run directory."""
    # Setup tag
    # ...
    
    resolved = resolve_input_path("@baseline")
    assert resolved.exists()
    assert "infer" in str(resolved)
```

#### F. Effort Estimate

- **Files to create:** 3 (run_tag_manager.py, tag_resolver.py, tag_cli.py)
- **Files to modify:** 5 (common_params.py, 4 CLI files)
- **Tests to write:** 4-5 test files
- **Time estimate:** 8-10 hours
- **Risk:** Medium (symlink handling, path resolution edge cases)

---

## Feature 3: Run Templates / Profiles

### Problem
Users must specify the same config flags repeatedly for common experiment patterns.

**Current UX:**
```bash
infer --model gpt-oss-20b-free --prompt thomas-simple --io json --input data.json
infer --model gpt-oss-20b-free --prompt thomas-simple --io json --input data2.json
infer --model gpt-oss-20b-free --prompt thomas-simple --io json --input data3.json
```

**Desired UX:**
```bash
# Save template from current run
infer --model gpt-oss-20b-free --prompt thomas-simple --io json --save-template my-experiment

# Reuse template
infer --from-template my-experiment --input data.json
infer --from-template my-experiment --input data2.json --override model.temperature=0.9

# List templates
llm-ensemble template list
llm-ensemble template list --cli infer

# Show template details
llm-ensemble template show my-experiment

# Remove template
llm-ensemble template remove my-experiment
```

### Benefits
- **Efficiency** - Reduce repetitive typing for common experiment patterns
- **Reproducibility** - Codify standard configurations as reusable profiles
- **Shareability** - Team members can share experiment configurations

### Implementation Strategy

#### A. Template Storage Design

**File format: YAML profiles**
```yaml
# .llm-ensemble/templates/infer/my-experiment.yaml
cli: infer
name: my-experiment
description: "Baseline GPT experiment with standard prompt"
created_at: "2025-01-28T14:30:22Z"

# Config references (same as CLI flags)
configs:
  model_cfg: gpt-oss-20b-free
  prompt_cfg: thomas-simple
  io_cfg: json
  retry_cfg: standard

# Default overrides (can be further overridden via --override)
overrides:
  model:
    temperature: 0.7
  prompt: {}
  io: {}

# Default CLI options
options:
  limit: null
  official: false
```

**Storage location:**
```
.llm-ensemble/templates/
├── infer/
│   ├── my-experiment.yaml
│   └── baseline.yaml
├── aggregate/
│   └── prod-ensemble.yaml
└── ingest/
    └── standard-ingest.yaml
```

**Why `.llm-ensemble/` in project root:**
- ✅ Standard hidden directory pattern (like `.git`, `.vscode`)
- ✅ Git-trackable for team sharing
- ✅ Separate from runtime artifacts
- ✅ Can add `.llm-ensemble/templates/.gitignore` for local templates

#### B. Architecture

```
libs/runtime/
├── template_manager.py   # NEW - Template CRUD operations
└── template_loader.py    # NEW - Load and merge template with CLI args

libs/cli/
└── template_cli.py       # NEW - llm-ensemble template subcommands

libs/schemas/
└── template_schema.py    # NEW - Pydantic schema for template files
```

#### C. Core Implementation

```python
# libs/schemas/template_schema.py
"""Schema for CLI run templates."""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TemplateConfigs(BaseModel):
    """Config references for a template."""
    # CLI-specific configs (some may be None depending on CLI)
    model_cfg: Optional[str] = None
    prompt_cfg: Optional[str] = None
    io_cfg: Optional[str] = None
    retry_cfg: Optional[str] = None
    ensemble_cfg: Optional[str] = None


class TemplateOptions(BaseModel):
    """Default CLI options for a template."""
    limit: Optional[int] = None
    official: bool = False
    notes: Optional[str] = None


class RunTemplate(BaseModel):
    """Reusable CLI run configuration template.
    
    Templates capture frequently-used combinations of configs and overrides,
    allowing users to run common experiments without repeating flags.
    """
    cli: str = Field(..., description="CLI name (infer, aggregate, etc.)")
    name: str = Field(..., description="Template name")
    description: Optional[str] = Field(None, description="Human-readable description")
    created_at: datetime = Field(default_factory=datetime.now)
    
    configs: TemplateConfigs = Field(..., description="Config references")
    overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Default config overrides (can be further overridden)"
    )
    options: TemplateOptions = Field(
        default_factory=TemplateOptions,
        description="Default CLI options"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "cli": "infer",
                "name": "my-experiment",
                "description": "Baseline GPT experiment",
                "configs": {
                    "model_cfg": "gpt-oss-20b-free",
                    "prompt_cfg": "thomas-simple",
                    "io_cfg": "json"
                },
                "overrides": {
                    "model": {"temperature": 0.7}
                },
                "options": {
                    "limit": None,
                    "official": False
                }
            }
        }
```

```python
# libs/runtime/template_manager.py
"""Manage reusable CLI run templates."""

from pathlib import Path
from typing import List, Optional
import yaml
from datetime import datetime

from llm_ensemble.libs.schemas.template_schema import RunTemplate
from llm_ensemble.libs.runtime.path_manager import PathManager


class TemplateManager:
    """Manages reusable run templates for all CLIs."""
    
    @staticmethod
    def get_templates_dir() -> Path:
        """Get the templates directory."""
        return PathManager.get_project_root() / ".llm-ensemble" / "templates"
    
    @staticmethod
    def get_cli_templates_dir(cli_name: str) -> Path:
        """Get templates directory for specific CLI."""
        return TemplateManager.get_templates_dir() / cli_name
    
    @staticmethod
    def save_template(template: RunTemplate, overwrite: bool = False) -> None:
        """Save a template to disk.
        
        Args:
            template: Template to save
            overwrite: If True, replace existing template
        
        Raises:
            ValueError: If template already exists and overwrite=False
        """
        cli_templates_dir = TemplateManager.get_cli_templates_dir(template.cli)
        cli_templates_dir.mkdir(parents=True, exist_ok=True)
        
        template_path = cli_templates_dir / f"{template.name}.yaml"
        
        if template_path.exists() and not overwrite:
            raise ValueError(
                f"Template '{template.name}' already exists for {template.cli}. "
                f"Use --overwrite to replace."
            )
        
        # Serialize to YAML
        with open(template_path, "w", encoding="utf-8") as f:
            yaml.dump(template.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def load_template(cli_name: str, template_name: str) -> RunTemplate:
        """Load a template from disk.
        
        Args:
            cli_name: CLI name (infer, aggregate, etc.)
            template_name: Template name
        
        Returns:
            Loaded template
        
        Raises:
            FileNotFoundError: If template doesn't exist
        """
        template_path = TemplateManager.get_cli_templates_dir(cli_name) / f"{template_name}.yaml"
        
        if not template_path.exists():
            # List available templates for helpful error
            available = TemplateManager.list_templates(cli_name)
            available_str = ", ".join(available) if available else "(none)"
            raise FileNotFoundError(
                f"Template not found: {template_name} (for {cli_name})\n"
                f"Available templates: {available_str}\n"
                f"Create templates with: {cli_name} --save-template {template_name} ..."
            )
        
        with open(template_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        return RunTemplate(**data)
    
    @staticmethod
    def list_templates(cli_name: Optional[str] = None) -> List[str]:
        """List available templates, optionally filtered by CLI.
        
        Args:
            cli_name: Only list templates for this CLI (None = all CLIs)
        
        Returns:
            List of template names
        """
        if cli_name:
            cli_dirs = [TemplateManager.get_cli_templates_dir(cli_name)]
        else:
            templates_root = TemplateManager.get_templates_dir()
            cli_dirs = [d for d in templates_root.iterdir() if d.is_dir()]
        
        templates = []
        for cli_dir in cli_dirs:
            if not cli_dir.exists():
                continue
            for template_file in cli_dir.glob("*.yaml"):
                templates.append(template_file.stem)
        
        return sorted(templates)
    
    @staticmethod
    def remove_template(cli_name: str, template_name: str) -> bool:
        """Remove a template.
        
        Args:
            cli_name: CLI name
            template_name: Template name
        
        Returns:
            True if removed, False if didn't exist
        """
        template_path = TemplateManager.get_cli_templates_dir(cli_name) / f"{template_name}.yaml"
        
        if not template_path.exists():
            return False
        
        template_path.unlink()
        return True
```

```python
# libs/runtime/template_loader.py
"""Load templates and merge with CLI arguments."""

from typing import Dict, Any, Optional
from llm_ensemble.libs.schemas.template_schema import RunTemplate
from llm_ensemble.libs.runtime.template_manager import TemplateManager


def load_and_merge_template(
    cli_name: str,
    template_name: str,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load template and merge with CLI-provided overrides.
    
    Template overrides serve as defaults, but can be further overridden
    by CLI --override flags.
    
    Args:
        cli_name: CLI name (infer, aggregate, etc.)
        template_name: Template to load
        cli_overrides: Additional overrides from --override flags
    
    Returns:
        Merged configuration dict with:
        - configs: Config names (model_cfg, prompt_cfg, etc.)
        - overrides: Merged overrides (template defaults + CLI overrides)
        - options: CLI options from template
    
    Raises:
        FileNotFoundError: If template doesn't exist
    """
    template = TemplateManager.load_template(cli_name, template_name)
    
    # Start with template's config overrides as base
    merged_overrides = dict(template.overrides)
    
    # Merge CLI-provided overrides (they take precedence)
    if cli_overrides:
        for prefix, overrides in cli_overrides.items():
            if prefix not in merged_overrides:
                merged_overrides[prefix] = {}
            merged_overrides[prefix].update(overrides)
    
    return {
        "configs": template.configs.model_dump(exclude_none=True),
        "overrides": merged_overrides,
        "options": template.options.model_dump(),
    }
```

#### D. CLI Integration

**1. Standalone template management CLI:**

```python
# src/llm_ensemble/template_cli.py
"""Template management CLI - llm-ensemble template subcommands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax

from llm_ensemble.libs.runtime.template_manager import TemplateManager

app = typer.Typer(help="Manage run templates")
console = Console()


@app.command("list")
def list_templates(
    cli: str = typer.Option(None, "--cli", help="Filter by CLI (infer, aggregate, etc.)"),
):
    """List all templates."""
    templates = TemplateManager.list_templates(cli_name=cli)
    
    if not templates:
        console.print("No templates found.")
        return
    
    console.print(f"[bold]Available templates{' for ' + cli if cli else ''}:[/bold]")
    for name in templates:
        console.print(f"  • {name}")


@app.command("show")
def show_template(
    cli: str = typer.Argument(..., help="CLI name (infer, aggregate, etc.)"),
    name: str = typer.Argument(..., help="Template name"),
):
    """Show template details."""
    try:
        template = TemplateManager.load_template(cli, name)
        
        # Display as formatted YAML
        import yaml
        template_yaml = yaml.dump(template.model_dump(), default_flow_style=False, sort_keys=False)
        syntax = Syntax(template_yaml, "yaml", theme="monokai", line_numbers=False)
        
        console.print(f"\n[bold cyan]Template: {cli}/{name}[/bold cyan]\n")
        console.print(syntax)
        
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("remove")
def remove_template(
    cli: str = typer.Argument(..., help="CLI name"),
    name: str = typer.Argument(..., help="Template name"),
):
    """Remove a template."""
    if TemplateManager.remove_template(cli, name):
        console.print(f"✓ Removed template [green]{cli}/{name}[/green]")
    else:
        console.print(f"[yellow]Template not found:[/yellow] {cli}/{name}")
        raise typer.Exit(1)
```

**2. Add --save-template and --from-template to CLIs:**

```python
# libs/cli/common_params.py (additions)
SaveTemplate = Annotated[
    Optional[str],
    typer.Option(
        "--save-template",
        help="Save this run's configuration as a reusable template"
    )
]

FromTemplate = Annotated[
    Optional[str],
    typer.Option(
        "--from-template",
        help="Load configuration from a template (can be overridden with other flags)"
    )
]
```

```python
# infer_cli.py (major changes)
from llm_ensemble.libs.cli.common_params import SaveTemplate, FromTemplate
from llm_ensemble.libs.runtime.template_loader import load_and_merge_template
from llm_ensemble.libs.runtime.template_manager import TemplateManager
from llm_ensemble.libs.schemas.template_schema import RunTemplate, TemplateConfigs, TemplateOptions

@app.command("infer")
def infer(
    # Template loading (mutually exclusive group ideally)
    from_template: FromTemplate = None,
    save_template: SaveTemplate = None,
    
    # Required parameters (now Optional if using template)
    io_cfg: Optional[str] = typer.Option(None, "--io-cfg", ...),
    model_cfg: Optional[str] = typer.Option(None, "--model-cfg", ...),
    prompt_cfg: Optional[str] = typer.Option(None, "--prompt-cfg", ...),
    
    # ... rest of params
):
    """Run LLM inference on judging samples..."""
    
    # Load from template if specified
    if from_template:
        template_data = load_and_merge_template(
            cli_name="infer",
            template_name=from_template,
            cli_overrides=parse_and_route_overrides(override) if override else None,
        )
        
        # Use template values as defaults, CLI flags override
        model_cfg = model_cfg or template_data["configs"].get("model_cfg")
        prompt_cfg = prompt_cfg or template_data["configs"].get("prompt_cfg")
        io_cfg = io_cfg or template_data["configs"].get("io_cfg")
        retry_cfg = retry_cfg or template_data["configs"].get("retry_cfg", "standard")
        
        # Merge template options
        limit = limit or template_data["options"].get("limit")
        official = official or template_data["options"].get("official", False)
        notes = notes or template_data["options"].get("notes")
        
        # Template overrides are already in template_data["overrides"]
        template_overrides = template_data["overrides"]
    else:
        template_overrides = {}
    
    # Validate required configs (even with template)
    if not model_cfg:
        raise typer.BadParameter("--model-cfg is required (or use --from-template)")
    if not prompt_cfg:
        raise typer.BadParameter("--prompt-cfg is required (or use --from-template)")
    if not io_cfg:
        raise typer.BadParameter("--io-cfg is required (or use --from-template)")
    
    # Load configurations
    model_config = load_model_config(model_cfg)
    prompt_config = load_prompt_config(prompt_cfg)
    retry_config = load_retry_config(retry_cfg)
    io_config = load_io_config(io_cfg, cli_name="infer")
    logging_config = load_logging_config(log_cfg or "standard")
    
    # Apply overrides (template overrides + CLI overrides)
    if override or template_overrides:
        cli_overrides = parse_and_route_overrides(override) if override else {}
        
        # Merge template and CLI overrides (CLI takes precedence)
        final_overrides = {}
        for prefix in ['model', 'prompt', 'io']:
            final_overrides[prefix] = {}
            final_overrides[prefix].update(template_overrides.get(prefix, {}))
            final_overrides[prefix].update(cli_overrides.get(prefix, {}))
        
        # Apply merged overrides
        if final_overrides['model']:
            model_config = apply_overrides(model_config, final_overrides['model'])
        if final_overrides['prompt']:
            prompt_config = apply_overrides(prompt_config, final_overrides['prompt'])
        if final_overrides['io']:
            io_config = apply_overrides(io_config, final_overrides['io'])
    
    # Run inference (existing logic)
    run_inference(...)
    
    # Save as template if requested
    if save_template:
        try:
            template = RunTemplate(
                cli="infer",
                name=save_template,
                description=notes or f"Template created from run {run_name}",
                configs=TemplateConfigs(
                    model_cfg=model_cfg,
                    prompt_cfg=prompt_cfg,
                    io_cfg=io_cfg,
                    retry_cfg=retry_cfg,
                ),
                overrides=final_overrides if (override or template_overrides) else {},
                options=TemplateOptions(
                    limit=limit,
                    official=official,
                    notes=notes,
                ),
            )
            TemplateManager.save_template(template)
            logger.info(f"Saved template as {save_template}")
        except Exception as e:
            logger.warning(f"Failed to save template: {e}")
```

#### E. Testing Strategy

```python
# tests/libs/runtime/test_template_manager.py
"""Tests for template management."""

def test_save_and_load_template(tmp_path):
    """Should save and load template."""
    template = RunTemplate(
        cli="infer",
        name="test-template",
        configs=TemplateConfigs(model_cfg="gpt-oss-20b-free"),
        overrides={},
        options=TemplateOptions(),
    )
    
    TemplateManager.save_template(template)
    loaded = TemplateManager.load_template("infer", "test-template")
    
    assert loaded.name == "test-template"
    assert loaded.configs.model_cfg == "gpt-oss-20b-free"


def test_load_and_merge_template():
    """Should merge template with CLI overrides."""
    # Create template with default temperature
    # ...
    
    merged = load_and_merge_template(
        "infer",
        "test-template",
        cli_overrides={"model": {"temperature": 0.9}},
    )
    
    # CLI override should win
    assert merged["overrides"]["model"]["temperature"] == 0.9
```

#### F. Effort Estimate

- **Files to create:** 4 (template_schema.py, template_manager.py, template_loader.py, template_cli.py)
- **Files to modify:** 6 (common_params.py, 4 CLI files, significant refactoring)
- **Tests to write:** 5-6 test files
- **Time estimate:** 12-16 hours
- **Risk:** High (complex merge logic, backward compatibility, many edge cases)

---

## Implementation Priority & Roadmap

### Phase 1: Display User Misses (Week 1) ⭐
**Why first:**
- Immediate user value (better discoverability)
- No state management needed
- Low risk, quick wins
- Teaches users the config-first design

**Deliverables:**
- ✅ Rich error messages for missing configs
- ✅ List available configs in error messages
- ✅ Usage examples in errors
- ✅ Works across all 4 CLIs

**Success Criteria:**
- User runs `infer` without flags → sees available models/prompts/io configs
- Error messages include example usage
- No change to existing functionality

---

### Phase 2: Run References / Shortcuts (Week 2-3)
**Why second:**
- Builds on Phase 1 (better errors if tag not found)
- Enables better workflows for multi-stage pipelines
- Moderate complexity (symlinks + path resolution)
- Unlocks shareability and reproducibility

**Deliverables:**
- ✅ `llm-ensemble tag` CLI with add/list/remove commands
- ✅ --tag flag for all CLIs (create tag after run)
- ✅ @tag resolution in --input paths
- ✅ Symlink-based tag storage in artifacts/runs/.tags/

**Success Criteria:**
- User can tag runs: `infer ... --tag baseline`
- User can reference: `aggregate --input @baseline`
- `llm-ensemble tag list` shows all tags
- Tags work across all 4 CLIs

---

### Phase 3: Run Templates / Profiles (Week 4-5)
**Why third:**
- Most complex (template storage, merge logic, CLI refactoring)
- Depends on good error messages from Phase 1
- Can reference tagged runs from Phase 2
- Requires significant CLI changes (optional params, template loading)

**Deliverables:**
- ✅ `llm-ensemble template` CLI with list/show/remove commands
- ✅ --save-template flag for all CLIs
- ✅ --from-template flag for all CLIs
- ✅ Template storage in .llm-ensemble/templates/
- ✅ Override merge logic (template defaults + CLI overrides)

**Success Criteria:**
- User can save: `infer ... --save-template my-experiment`
- User can reuse: `infer --from-template my-experiment --input data.json`
- Overrides work: `infer --from-template baseline --override model.temperature=0.9`
- `llm-ensemble template list` shows available templates
- Templates work across all 4 CLIs

---

## Cross-Cutting Concerns

### 1. Main CLI Entrypoint

Currently each CLI is standalone. Need a main entrypoint for tag/template commands:

```python
# src/llm_ensemble/cli.py (NEW)
"""Main CLI entrypoint for llm-ensemble."""

import typer
from llm_ensemble import ingest_cli, infer_cli, aggregate_cli, evaluate_cli
from llm_ensemble.tag_cli import app as tag_app
from llm_ensemble.template_cli import app as template_app

app = typer.Typer(
    help="LLM Ensemble - CLI-first LLM relevance judging system",
    pretty_exceptions_enable=False,
)

# Register subcommands
app.add_typer(ingest_cli.app, name="ingest")
app.add_typer(infer_cli.app, name="infer")
app.add_typer(aggregate_cli.app, name="aggregate")
app.add_typer(evaluate_cli.app, name="evaluate")
app.add_typer(tag_app, name="tag")
app.add_typer(template_app, name="template")

if __name__ == "__main__":
    app()
```

**Update pyproject.toml:**
```toml
[project.scripts]
llm-ensemble = "llm_ensemble.cli:app"
ingest = "llm_ensemble.ingest_cli:app"
infer = "llm_ensemble.infer_cli:app"
aggregate = "llm_ensemble.aggregate_cli:app"
evaluate = "llm_ensemble.evaluate_cli:app"
```

### 2. Documentation Updates

Each phase needs corresponding docs:
- **Phase 1:** Update README with new error message examples
- **Phase 2:** Add "Working with Tags" section to docs
- **Phase 3:** Add "Using Templates" tutorial with examples

### 3. Backward Compatibility

All features are **additive** - no breaking changes:
- Phase 1: Only changes error messages
- Phase 2: Adds new flags (--tag) and @tag syntax
- Phase 3: Adds new flags (--save-template, --from-template)

Existing workflows continue to work unchanged.

---

## Risk Assessment

| Risk | Phase | Mitigation |
|------|-------|------------|
| Symlink support on Windows | 2 | Test on Windows, fallback to JSON registry if needed |
| Template merge logic bugs | 3 | Comprehensive unit tests, start with simple merge rules |
| Config override conflicts | 3 | Clear precedence rules (CLI > template > config defaults) |
| Path resolution edge cases | 2 | Test with relative/absolute paths, missing targets |
| CLI flag explosion | 3 | Keep optional, document clearly, use mutually exclusive groups |

---

## Next Steps

**Start with Phase 1** (Display User Misses) because:
1. ✅ Immediate user value
2. ✅ Low risk
3. ✅ Quick to implement (1 week)
4. ✅ Foundation for better errors in Phase 2 & 3

**Recommended first task:**
Create `libs/cli/validation_callbacks.py` and `libs/cli/error_messages.py` with rich error formatting for missing model/prompt/io configs in the infer CLI.

Once Phase 1 is working and tested, we can move to Phase 2 (tags) which builds on the improved error messages.
