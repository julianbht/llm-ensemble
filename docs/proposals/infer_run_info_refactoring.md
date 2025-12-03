# InferRunInfo Refactoring Proposal

## Problem Statement

The current `InferRunInfo` schema mixes multiple concerns:
- **Config names** (unresolved references like `"thomas-simple"`)
- **Full config objects** (resolved configs like `ModelConfig`)
- **CLI parameters** (user inputs like `input_run_name`, `start_idx`, `end_idx`)
- **Runtime metadata** (git state, run identification)

This creates confusion:
1. **Duplication**: `prompt_name` and `parser_name` are stored in `InferRunInfo`, then passed separately to `writer.open()`, then stored again in `config_names` dict for ORM mapping
2. **Inconsistency**: Some configs are stored as objects (`model_cfg`, `retry_config`), others only as names (`prompt_name`, `parser_name`)
3. **Resolution gap**: Orchestrator resolves config names → adapters, but this resolution isn't captured structurally
4. **Mixed layers**: Domain service receives `InferRunInfo` (infrastructure) but shouldn't need all that detail

## Proposed Structure

### 1. InferCliArgs - What the user typed

```python
@dataclass(frozen=True)
class InferCliArgs:
    """CLI arguments for infer command - user inputs only.
    
    This captures exactly what the user typed at the command line.
    No resolution, no computation, no git metadata.
    """
    # Config references (unresolved)
    model_config_name: str
    prompt_name: str
    parser_name: str
    retry_config_name: str
    io_config_name: str
    
    # Input data
    input_run_name: str  # Which ingest run to read
    
    # Processing range
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    
    # Run control
    run_name: Optional[str] = None  # Custom run name (auto-generated if None)
    official: bool = False
    tag: Optional[str] = None
    notes: Optional[str] = None
```

**Location**: `src/llm_ensemble/infer/schemas/cli_args.py`

---

### 2. InferConfig - Resolved configuration bundle

```python
@dataclass(frozen=True)
class InferConfig:
    """Fully resolved configuration for an infer run.
    
    All config objects are loaded and validated. This is what the
    domain service needs to execute the pipeline.
    """
    # Resolved configs (loaded from files)
    model_config: ModelConfig
    retry_config: RetryConfig
    io_config: IOConfig
    
    # Adapter specs (resolved from registries)
    prompt_spec: PromptSpec
    parser_spec: ParserSpec
    
    # Config names (for provenance tracking)
    config_names: ConfigNames
```

Where:

```python
@dataclass(frozen=True)
class PromptSpec:
    """Resolved prompt adapter specification."""
    name: str  # Registry name (e.g., "thomas-simple")
    template_id: UUID  # Deterministic UUID from name
    template_text: str  # Loaded template text

@dataclass(frozen=True)
class ParserSpec:
    """Resolved parser adapter specification."""
    name: str  # Registry name (e.g., "thomas-simple")
    parser_id: UUID  # Deterministic UUID from name

@dataclass(frozen=True)
class ConfigNames:
    """Config names for provenance tracking."""
    model_config: str
    prompt: str
    parser: str
    retry: str
    io: str
```

**Location**: `src/llm_ensemble/infer/schemas/config.py`

---

### 3. InferRunContext - Complete runtime context

```python
class InferRunContext(BaseModel):
    """Complete runtime context for an infer run.
    
    Combines resolved configuration with runtime metadata.
    This is the immutable context that gets persisted and attached
    to domain objects for full provenance.
    """
    # Run identification (from base RunInfo)
    run_name: str
    run_type: RunType
    cli_name: Literal["infer"] = "infer"
    
    # Git metadata (auto-captured)
    git_sha: str
    git_branch: str
    git_clean: bool
    
    # User context
    notes: Optional[str] = None
    
    # Input specification
    input_run_name: str  # Which ingest run was processed
    start_idx: int  # Actual start index (resolved from Optional to int)
    end_idx: int  # Actual end index (resolved from Optional to int)
    
    # Configuration provenance (names only - configs stored separately)
    config_names: ConfigNames
    
    model_config = ConfigDict(frozen=True)
```

**Location**: `src/llm_ensemble/infer/schemas/run_context.py`

**Why separate from configs?**
- Configs are bulky (contain full objects) - don't want to serialize with every judgement
- Run context is lightweight metadata that can be embedded in domain objects
- Configs stored once in manifest, run context provides references

---

### 4. Domain Service Signature Changes

```python
class InferenceService:
    def run_inference(
        self,
        input_run_name: str,
        config: InferConfig,  # ← All configs bundled
        run_context: InferRunContext,  # ← Lightweight context
        run_dir: Path,
    ) -> InferRunSummary:
        """Execute inference pipeline."""
        
        # No need to extract config names - already in run_context
        # No need to pass prompt_name/parser_name separately - in config
        
        with self.judgement_writer.open(
            run_dir=run_dir,
            run_context=run_context,
            config=config,  # Writer extracts what it needs
        ) as writer:
            # ... inference loop
```

**Benefits:**
- Domain service receives clean, cohesive bundles
- No more passing the same data in 3 different forms
- Writer can extract what it needs from config bundle

---

### 5. Writer Port Changes

```python
class JudgementWriter(ABC):
    @abstractmethod
    def open(
        self,
        run_dir: Path,
        run_context: InferRunContext,
        config: InferConfig,
        normalized_dataset: NormalizedDataset,
    ) -> ContextManager[JudgementWriter]:
        """Open writer for streaming writes.
        
        Args:
            run_dir: Output directory
            run_context: Lightweight runtime context
            config: Full configuration bundle (extract what you need)
            normalized_dataset: Input dataset for fingerprinting
        """
```

**SQL Repository Implementation:**
```python
def open(self, run_dir, run_context, config, normalized_dataset):
    # Extract what we need
    prompt_spec = config.prompt_spec
    parser_spec = config.parser_spec
    model_config = config.model_config
    
    # Initialize metadata
    self._initialize_run_metadata(
        run_context=run_context,
        prompt_spec=prompt_spec,
        parser_spec=parser_spec,
        model_config=model_config,
    )
```

---

### 6. Orchestrator Flow

```python
def run_inference(cli_args: InferCliArgs) -> None:
    """Orchestrate inference run from CLI arguments."""
    
    # 1. LOAD CONFIGS (resolve names → objects)
    model_config = load_model_config(cli_args.model_config_name)
    retry_config = load_retry_config(cli_args.retry_config_name)
    io_config = load_io_config(cli_args.io_config_name)
    
    # 2. BUILD ADAPTER SPECS (resolve names → specs with UUIDs)
    prompt_spec = PromptAdapterBuilder.get_spec(cli_args.prompt_name)
    parser_spec = ParserAdapterBuilder.get_spec(cli_args.parser_name)
    
    # 3. BUNDLE INTO CONFIG
    config = InferConfig(
        model_config=model_config,
        retry_config=retry_config,
        io_config=io_config,
        prompt_spec=prompt_spec,
        parser_spec=parser_spec,
        config_names=ConfigNames(
            model_config=cli_args.model_config_name,
            prompt=cli_args.prompt_name,
            parser=cli_args.parser_name,
            retry=cli_args.retry_config_name,
            io=cli_args.io_config_name,
        ),
    )
    
    # 4. GENERATE RUN METADATA
    run_name = cli_args.run_name or generate_run_name([
        model_config.name_hint,
        prompt_spec.name,
        parser_spec.name,
    ])
    git_info = get_git_info()
    
    # 5. READ INPUT TO RESOLVE INDICES
    reader = io_config.get_reader()
    normalized_dataset = reader.read(cli_args.input_run_name)
    start_idx = cli_args.start_idx or 0
    end_idx = cli_args.end_idx or len(normalized_dataset.samples)
    
    # 6. BUILD RUN CONTEXT (lightweight, serializable)
    run_context = InferRunContext(
        run_name=run_name,
        run_type=RunType.OFFICIAL if cli_args.official else RunType.TEST,
        git_sha=git_info["git_sha"],
        git_branch=git_info["git_branch"],
        git_clean=git_info["git_clean"],
        notes=cli_args.notes,
        input_run_name=cli_args.input_run_name,
        start_idx=start_idx,
        end_idx=end_idx,
        config_names=config.config_names,
    )
    
    # 7. CREATE RUN DIRECTORY
    run_dir = run_context.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 8. INSTANTIATE ADAPTERS
    prompt_builder = PromptAdapterBuilder.build(prompt_spec)
    parser = ParserAdapterBuilder.build(parser_spec)
    provider = model_config.get_provider()
    reader = io_config.get_reader()
    writer = io_config.get_writer()
    
    # 9. BUILD DOMAIN SERVICE
    service = InferenceService(
        example_reader=reader,
        judgement_writer=writer,
        prompt_builder=prompt_builder,
        llm_provider=provider,
        response_parser=parser,
    )
    
    # 10. RUN INFERENCE
    summary = service.run_inference(
        input_run_name=cli_args.input_run_name,
        config=config,
        run_context=run_context,
        run_dir=run_dir,
    )
    
    # 11. PERSIST MANIFESTS
    write_config_manifest(run_dir / "config.json", config)
    write_context_manifest(run_dir / "run_context.json", run_context)
    write_summary(run_dir / "summary.json", summary)
```

---

## Migration Path

### Phase 1: Add new schemas (non-breaking)
1. Create `InferCliArgs`, `InferConfig`, `InferRunContext` schemas
2. Create `PromptSpec`, `ParserSpec`, `ConfigNames` helper dataclasses
3. Add builder methods: `PromptAdapterBuilder.get_spec()`, `ParserAdapterBuilder.get_spec()`

### Phase 2: Update orchestrator (internal only)
1. Refactor `run_inference()` to use new flow
2. Keep `InferRunInfo` for now - map from new schemas to old

### Phase 3: Update domain service & ports (interface changes)
1. Change `InferenceService.run_inference()` signature
2. Change `JudgementWriter.open()` signature
3. Update all adapter implementations

### Phase 4: Remove old schema
1. Delete `InferRunInfo`
2. Update all imports

---

## Benefits

### ✅ Clear separation of concerns
- **CLI args** = user input
- **Config** = resolved configuration
- **Run context** = runtime metadata

### ✅ No duplication
- Config names stored once in `ConfigNames`
- No more passing `prompt_name`, `parser_name` separately

### ✅ Explicit resolution
- `PromptSpec` and `ParserSpec` capture the resolution step
- UUIDs computed at resolution time, not deep in mappers

### ✅ Better testability
- Can construct `InferConfig` directly in tests
- Can mock specs without full registry

### ✅ Cleaner domain layer
- Domain service receives clean bundles
- No infrastructure leakage (git metadata stays in context)

### ✅ Consistent with architecture
- Orchestrator layer: CLI args → Config + Context
- Domain layer: Config + Context → Results
- Adapter layer: Specs → Implementations

---

## Open Questions

1. **Should `InferRunContext` extend `RunInfo`?**
   - Pro: Reuses base class, consistent with current design
   - Con: Base `RunInfo` might not fit the new structure
   - **Recommendation**: Start fresh, migrate common fields

2. **Where do we store full config objects?**
   - Option A: Separate `config.json` manifest (my recommendation)
   - Option B: Embed in `run_context.json` (current approach)
   - **Recommendation**: Separate files - run_context should be lightweight

3. **Should adapters receive `PromptSpec`/`ParserSpec` in constructors?**
   - Current: Adapters receive just `name: str`
   - Proposed: Adapters receive full spec with UUID
   - **Recommendation**: Yes - adapters should know their identity upfront

4. **Do we need `InferCliArgs` as a formal schema?**
   - Could just pass individual params to orchestrator
   - **Recommendation**: Yes - makes CLI → orchestrator boundary explicit

---

## Example File Outputs

### `artifacts/runs/infer/test/20250103_142859_gpt4-thomas/run_context.json`
```json
{
  "run_name": "20250103_142859_gpt4-thomas",
  "run_type": "test",
  "cli_name": "infer",
  "git_sha": "abc123...",
  "git_branch": "main",
  "git_clean": true,
  "input_run_name": "my_ingest_run",
  "start_idx": 0,
  "end_idx": 100,
  "config_names": {
    "model_config": "gpt-oss-20b",
    "prompt": "thomas-simple",
    "parser": "thomas-simple",
    "retry": "standard",
    "io": "json"
  }
}
```

### `artifacts/runs/infer/test/20250103_142859_gpt4-thomas/config.json`
```json
{
  "model_config": {
    "name": "gpt-oss-20b",
    "model_id": "meta-llama/llama-3.1-8b-instruct",
    "provider": "openrouter",
    "temperature": 0.7,
    ...
  },
  "prompt_spec": {
    "name": "thomas-simple",
    "template_id": "uuid-from-name",
    "template_text": "Query: {{query}}\n..."
  },
  "parser_spec": {
    "name": "thomas-simple",
    "parser_id": "uuid-from-name"
  },
  "retry_config": {...},
  "io_config": {...}
}
```

---

## Summary

This refactoring cleanly separates:
1. **User input** (CLI args)
2. **Resolved configuration** (loaded configs + adapter specs)
3. **Runtime context** (git state + run metadata)

The orchestrator becomes the "resolution layer" that transforms CLI args → config + context, then hands clean bundles to the domain service. No more passing the same data in multiple forms, no more mixing concerns.
