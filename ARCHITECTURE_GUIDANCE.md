# Runtime Context Pattern - Architecture Decision

## Problem Statement

The composition root (`dependency_configurator.py`) is being polluted with runtime infrastructure concerns (logging config, run directory overrides for tests). This violates Single Responsibility Principle and makes testing awkward.

## Root Cause

Conflating two distinct concerns:
1. **Adapter wiring** (compile-time): Which implementations to use
2. **Runtime configuration** (runtime): Where to put files, how to log, test vs. production

## Solution: Runtime Context Pattern

Separate runtime infrastructure configuration from adapter wiring using a `RuntimeContext` object.

### Architecture

```
┌──────────────────────────────────────────────────────┐
│ CLI (Driving Adapter)                                │
│                                                       │
│  1. Build adapters via DependencyConfigurator        │
│     └──> Returns: InferenceApplication               │
│                                                       │
│  2. Build runtime context                            │
│     └──> Returns: RuntimeContext                     │
│                                                       │
│  3. Execute: application.run_inference(context)      │
└──────────────────────────────────────────────────────┘
```

### Implementation Pattern

#### 1. RuntimeContext Entity (Domain)

```python
# src/llm_ensemble/infer/domain/entities/runtime_context.py

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from llm_ensemble.libs.schemas.logging_config import LoggingConfig

@dataclass(frozen=True)
class RuntimeContext:
    """Runtime infrastructure configuration for inference execution.

    Encapsulates all runtime concerns that are NOT about adapter selection:
    - Logging configuration
    - Run directory overrides (for testing)
    - Environment type (test vs. production)
    - Other runtime metadata

    This is separate from adapter wiring (composition root responsibility).
    """

    logging_config: LoggingConfig

    # Run directory override (None = auto-generate from run_name)
    run_dir_override: Optional[Path] = None

    # For testing: skip certain infrastructure steps
    skip_git_checks: bool = False

    @classmethod
    def for_production(cls, logging_config: LoggingConfig) -> "RuntimeContext":
        """Create production runtime context."""
        return cls(
            logging_config=logging_config,
            run_dir_override=None,
            skip_git_checks=False,
        )

    @classmethod
    def for_testing(
        cls,
        run_dir: Path,
        logging_config: Optional[LoggingConfig] = None,
    ) -> "RuntimeContext":
        """Create test runtime context with overrides."""
        if logging_config is None:
            # Default test logging: no file output, JSON format
            logging_config = LoggingConfig(
                pretty_print=False,
                save_logs=False,
                console_level="INFO",
                file_level="DEBUG",
            )

        return cls(
            logging_config=logging_config,
            run_dir_override=run_dir,
            skip_git_checks=True,
        )
```

#### 2. Updated InferenceApplication

```python
# src/llm_ensemble/infer/application/inference_application.py

class InferenceApplication(ForRunningInference):
    """Application use case for coordinating LLM inference pipeline."""

    def __init__(
        self,
        input_port: InputPort,
        output_port: OutputPort,
        prompt_builder: PromptBuilderPort,
        llm_provider: LLMProviderPort,
        response_parser: ResponseParserPort,
        # REMOVED: logging_config parameter
    ):
        """Initialize inference use case with port dependencies.

        Note: Logging and run directory configuration now passed via RuntimeContext
        in run_inference() method, not during construction.
        """
        self.input_port = input_port
        self.output_port = output_port
        self.prompt_builder = prompt_builder
        self.llm_provider = llm_provider
        self.response_parser = response_parser

    def run_inference(
        self,
        context: RuntimeContext,  # NEW: Runtime configuration
        input_run_name: str,
        start_idx: Optional[int],
        end_idx: Optional[int],
        run_name: Optional[str],
        official: bool,
        notes: Optional[str],
        tag: Optional[str],
    ) -> InferRunSummary:
        """Execute the complete inference backend with infrastructure setup.

        Args:
            context: Runtime configuration (logging, run dir overrides, etc.)
            ... (other args unchanged)
        """
        start_time = datetime.now()

        # Generate run name
        run_name = self._generate_run_name(run_name)

        # Create run directory (respecting context override)
        if context.run_dir_override:
            run_dir = context.run_dir_override
            run_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_dir = self._create_run_directory(run_name, official, tag)

        # Setup logging from context
        logger = self._setup_logging(run_name, run_dir, context.logging_config)

        # ... rest of implementation unchanged
```

#### 3. Cleaned-up DependencyConfigurator (Composition Root)

```python
# src/llm_ensemble/infer/startup/dependency_configurator.py

def build_application(
    provider_name: str,
    io_name: str,
    prompt_template_name: str,
    model_config_name: str,
    retry_config_name: str,
    # REMOVED: logging_config parameter
) -> ForRunningInference:
    """Build and wire the inference application hexagon.

    Composition root - ONLY handles adapter wiring.
    Runtime configuration (logging, run dirs) now handled via RuntimeContext.
    """
    # Load configuration from YAML files
    model_cfg = load_model_config(model_config_name)
    retry_cfg = load_retry_config(retry_config_name)

    # Build application hexagon with loaded configs
    return _build_application_hexagon(
        provider_name=provider_name,
        io_name=io_name,
        prompt_template_name=prompt_template_name,
        model_cfg=model_cfg,
        retry_cfg=retry_cfg,
        # REMOVED: logging_cfg
    )
```

#### 4. Updated CLI (Driving Adapter)

```python
# src/llm_ensemble/infer/adapters/driving/infer_cli.py

def infer(
    model_cfg: ModelCfg,
    provider: Provider,
    prompt_template: PromptTemplate,
    io_cfg: InferIoCfg,
    input_run_name: InferIngestRunInput,
    # ... other params
):
    """Run LLM inference on judging examples."""

    # Step 1: Build application (adapter wiring only)
    application = build_application(
        provider_name=provider,
        io_name=io_cfg,
        prompt_template_name=prompt_template,
        model_config_name=model_cfg,
        retry_config_name=retry_cfg,
    )

    # Step 2: Build runtime context (infrastructure configuration)
    from llm_ensemble.infer.startup.config_loader import load_logging_config
    logging_config = load_logging_config("observability")
    context = RuntimeContext.for_production(logging_config)

    # Step 3: Run application with context
    application.run_inference(
        context=context,
        input_run_name=input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        run_name=run_name,
        official=official,
        notes=notes,
        tag=tag,
    )
```

#### 5. Test Usage

```python
# tests/infer/test_inference_application.py

def test_inference_with_custom_logging(tmp_path):
    """Tests can provide custom RuntimeContext."""

    # Build application with real/mock adapters
    application = build_application(...)

    # Create test runtime context with overrides
    test_run_dir = tmp_path / "test_run"
    context = RuntimeContext.for_testing(
        run_dir=test_run_dir,
        logging_config=LoggingConfig(
            pretty_print=False,
            save_logs=False,  # Don't pollute test output
        ),
    )

    # Execute with test context
    summary = application.run_inference(
        context=context,
        input_run_name="test_input",
        # ... other params
    )

    # Verify output in expected test directory
    assert (test_run_dir / "summary.json").exists()
```

## Comparison with BlueZone Reference

Your BlueZone reference has similar separation:

- `DependencyConfigurator`: Wires adapters (driven + driving)
- `BlueZoneInitializer.init()`: Post-construction runtime initialization

The RuntimeContext pattern achieves similar separation but is more explicit and testable.

## Benefits

1. **Clean composition root**: Only handles adapter wiring
2. **Testable**: Tests provide test context, production uses production context
3. **Explicit**: All runtime configuration visible via RuntimeContext
4. **Flexible**: Easy to add new runtime concerns without polluting composition root
5. **Aligns with principles**: Separates concerns, explicit over implicit

## Migration Path

1. Create `RuntimeContext` entity in domain
2. Update `InferenceApplication.run_inference()` to accept context parameter
3. Remove `logging_config` from `InferenceApplication.__init__()`
4. Update `dependency_configurator` to remove logging_config parameter
5. Update CLI to build context and pass to application
6. Update tests to use `RuntimeContext.for_testing()`

## Alternative Considered: Builder Pattern

Could use a builder pattern for application construction:

```python
application = (
    InferenceApplicationBuilder()
    .with_adapters(input_port, output_port, ...)
    .with_runtime_context(context)
    .build()
)
```

**Rejected because**: Adds complexity without clear benefit. The RuntimeContext parameter pattern is simpler and more explicit.

## Relation to Spring's application.properties

This pattern is analogous to Spring's separation:

- **Spring Bean Configuration** → Your `dependency_configurator.py`
- **application.properties** → Your `RuntimeContext`

Spring separates:
- What beans to create (configuration)
- How to configure those beans at runtime (properties)

You're doing the same:
- What adapters to use (dependency_configurator)
- How to configure runtime infrastructure (RuntimeContext)
