# Infer CLI - Architecture & Design Patterns

The **infer** CLI runs LLM inference on judging examples and outputs structured judgements.

This document focuses on the architectural patterns and design principles used in this CLI, which serve as a reference for other CLIs in the project.

## Core Architectural Patterns

### 1. Hexagonal Architecture (Ports & Adapters)

**Strict separation of concerns through layered architecture:**

```
CLI Layer → Orchestrator Layer → Domain Layer → Ports Layer → Adapters Layer
```

**Key principle:** Domain logic depends only on port abstractions (ABCs), never on concrete implementations. This enables:
- Testing domain logic in complete isolation (no APIs, no files, no external dependencies)
- Swapping implementations via configuration (different providers, formats, parsers)
- Independent evolution of layers

**Layers:**

1. **CLI Layer** (`infer_cli.py`)
   - Pure wiring: argument parsing → orchestrator invocation
   - Zero business logic
   - Pattern: Typer for CLI framework

2. **Orchestrator Layer** (`orchestrator.py`)
   - Infrastructure coordination: run management, logging, manifests
   - Loads YAML configurations
   - Instantiates adapters dynamically from config (configuration-driven)
   - Delegates to domain service
   - Pattern: Orchestrator pattern

3. **Domain Layer** (`domain/inference_service.py`)
   - **Pure business logic** - the "how" of the inference pipeline
   - Coordinates: read → build prompt → infer → parse → write (streaming loop)
   - Depends ONLY on port abstractions (ABCs)
   - Zero I/O, zero infrastructure knowledge
   - Pattern: Domain service

4. **Ports Layer** (`ports/`)
   - Abstract base classes defining infrastructure contracts
   - Examples: `LLMProvider`, `ExampleReader`, `JudgementWriter`, `PromptBuilder`, `ResponseParser`
   - Pattern: Dependency inversion principle

5. **Adapters Layer** (`adapters/`)
   - Concrete implementations of ports
   - Handle all I/O, APIs, external systems
   - Organized by concern: `io/`, `providers/`, `prompts/`, `parsers/`
   - Pattern: Adapter pattern

### 2. Streaming with Immediate Persistence

**Problem:** Batch processing accumulates data in memory and loses partial progress on errors.

**Solution:** Process one item at a time with immediate disk writes.

**Pattern: Context Manager + Streaming Loop**

**Benefits:**
- ✅ **Fault tolerance** - partial progress preserved on errors
- ✅ **Live progress** - see results as they complete (logging callback invoked immediately)
- ✅ **Memory efficient** - no accumulation of all results
- ✅ **Simple control flow** - clean sequential loop, no parallel list tracking

**Key pattern: Context manager for resource lifecycle**

```python
class JudgementWriter(ABC):
    def open(self, run_dir: Path) -> "JudgementWriter":
        """Prepare for streaming writes."""
        pass

    def write_one(self, judgement: LLMJudgement) -> None:
        """Write single judgement immediately."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    def __enter__(self) -> "JudgementWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
```

### 3. Dynamic Adapter Instantiation

**Problem:** How to instantiate the right adapter based on configuration?

**Solution:** Config objects dynamically load adapters using module/class paths specified in YAML files.

**Rule:** When adding new adapters, create the adapter class and specify its module/class path in config files.

### 4. Explicit Configuration Over Implicit Defaults

**Principle:** Make all behavior visible and configurable. Errors over silent fallbacks.

**Examples:**
- ✅ All adapters explicitly specified in YAML configs
- ✅ CLI flags reference config names: `--model gpt-oss-20b` loads `configs/models/gpt-oss-20b.yaml`
- ✅ Configuration files bundle related concerns (prompt config bundles builder + parser)
- ✅ Missing config → **ValidationError** with clear message (not silent default)

**Rationale:** Transparency and predictability. Users understand what's happening and can adjust behavior by modifying configs.

## Domain Service Data Flow

**Streaming loop in `InferenceService.run_inference()`:**

```
1. Read all samples (samples are small metadata objects)
2. Open writer context manager
3. For each sample:
   ├─ Build prompt (PromptBuilder port)
   ├─ Infer (LLMProvider port) ← Simple synchronous call
   ├─ Parse response (ResponseParser port)
   ├─ Create LLMJudgement
   ├─ Invoke callback (live logging)
   └─ Write immediately to disk (JudgementWriter port) ← Persistence!
4. Close writer (automatic via context manager)
5. Calculate summary statistics
6. Return InferRunSummary
```

**Key characteristics:**
- Clean sequential flow (no parallel lists, no order tracking)
- Each judgement persisted before processing next (fault tolerance)
- Callback invoked immediately (live progress visibility)
- Simple error handling - exception at any point preserves previous work

## Key Takeaways

1. **Hexagonal architecture** enables testable, swappable components
2. **Streaming with immediate persistence** provides fault tolerance and live progress
3. **Context managers** handle resource lifecycle cleanly (files, connections)
4. **Simple interfaces** (avoid iterators/async unless needed) reduce complexity
5. **Dynamic adapter instantiation** enables configuration-driven behavior
6. **Explicit configuration** makes system transparent and predictable
7. **Domain service** coordinates pure business logic using only port abstractions

These patterns ensure the CLI is maintainable, testable, and extensible for research experimentation.
