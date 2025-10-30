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
   - Instantiates adapters via factories (configuration-driven)
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

```python
# InferenceService.run_inference()
with self.judgement_writer.open(run_dir) as writer:
    for sample in samples:
        # 1. Build prompt
        request = self.prompt_builder.build(sample)

        # 2. Infer (simple synchronous call)
        response = self.llm_provider.infer(request.prompt, model_config)

        # 3. Parse response
        score = self.response_parser.parse(response.raw_response)

        # 4. Create judgement
        judgement = LLMJudgement(sample, request, response, score, run_info)

        # 5. Log immediately (live progress!)
        if on_judgement:
            on_judgement(judgement)

        # 6. Persist immediately (fault tolerance!)
        writer.write_one(judgement)

        # If error occurs here ↑, all previous judgements are already saved
```

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

**NDJSON implementation:**
- Opens file once (`open()`)
- Writes + flushes each line immediately (`write_one()`)
- Auto-closes on context exit (`close()`)

**For formats requiring batching (e.g., Parquet):**
Adapter can buffer internally transparently to caller - the interface remains streaming.

### 3. Simple, Synchronous Interfaces (Avoid Premature Abstraction)

**Anti-pattern:** Iterator-based provider interface for "future batching support"

```python
# ❌ Old: Complex iterator interface
def infer(
    sample_prompt_pairs: Iterator[tuple[Sample, str]],
    config: ModelConfig
) -> Iterator[tuple[Sample, LLMResponse]]:
    for sample, prompt in sample_prompt_pairs:
        yield (sample, self._call_api(prompt))

# Usage in service required wrapping/unwrapping:
sample_prompt_pair = iter([(sample, prompt)])  # Wrap single item
for _, response in provider.infer(sample_prompt_pair, config):  # Unwrap
    # ... only ever get ONE response anyway
```

**Pattern:** YAGNI - start simple, complexify only when needed

```python
# ✅ New: Simple synchronous interface
def infer(self, prompt: str, model_config: ModelConfig) -> LLMResponse:
    """Run inference on a single prompt."""
    response = self._call_api(prompt)
    return response

# Usage in service is direct:
response = self.llm_provider.infer(request.prompt, model_config)  # Clean!
```

**Benefits:**
- ✅ **Easier to understand** - no iterator ceremony
- ✅ **Easier to debug** - straightforward call/return
- ✅ **Easier to test** - mock returns a value, not a generator

**If batching is needed later:**
- Keep the simple interface for callers
- Implement batching **inside the adapter** transparently (internal buffering)
- Or create a separate `BatchLLMProvider` port if truly needed

**Principle:** Interface complexity should match actual use cases, not imagined future needs.

### 4. Factory Pattern for Configuration-Driven Adapter Selection

**Problem:** How to instantiate the right adapter based on configuration?

**Solution:** Factory functions that read config and return concrete implementations.

**Pattern:**

```python
# adapters/provider_factory.py
def get_provider(model_config: ModelConfig) -> LLMProvider:
    """Instantiate provider based on config.provider field."""
    if model_config.provider == "openrouter":
        return OpenRouterAdapter(api_key=os.getenv("OPENROUTER_API_KEY"))
    elif model_config.provider == "ollama":
        return OllamaAdapter(base_url=os.getenv("OLLAMA_BASE_URL"))
    elif model_config.provider == "huggingface":
        return HuggingFaceAdapter(token=os.getenv("HF_TOKEN"))
    else:
        raise ValueError(f"Unknown provider: {model_config.provider}")
```

**Usage in orchestrator:**

```python
# Load config
model_config = load_model_config(model_id)  # Reads configs/models/{model_id}.yaml

# Factory instantiates the right adapter
provider = get_provider(model_config)

# Domain service uses only the port abstraction
service = InferenceService(provider=provider, ...)  # provider: LLMProvider (ABC)
```

**Factories in this CLI:**
- `provider_factory.py` - LLM providers (OpenRouter, Ollama, HF)
- `io_factory.py` - Readers and writers (NDJSON, Parquet)
- `prompt_builder_factory.py` - Prompt builders (Jinja2)
- `response_parser_factory.py` - Response parsers (JSON, regex)

**Benefits:**
- ✅ **Configuration-driven** - behavior changes via YAML, not code
- ✅ **Testable** - factories can return test doubles
- ✅ **Extensible** - add new adapters by updating factory + config

**Rule:** When adding new adapters, ALWAYS update the corresponding factory.

### 5. Explicit Configuration Over Implicit Defaults

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

## Extending the Infer CLI

### Adding a New Provider

1. Create `adapters/providers/my_provider_adapter.py` implementing `LLMProvider` port
2. Update `adapters/provider_factory.py` to handle `provider: my_provider`
3. Add model configs to `configs/models/` with `provider: my_provider`

**LLMProvider contract:**
```python
@abstractmethod
def infer(self, prompt: str, model_config: ModelConfig) -> LLMResponse:
    """Single prompt → single response (simple synchronous call)."""
    pass
```

### Adding a New I/O Format

1. Create reader implementing `ExampleReader` port
2. Create writer implementing `JudgementWriter` port (with context manager support)
3. Update `adapters/io_factory.py`
4. Add config to `configs/io/my_format.yaml`

**JudgementWriter contract:**
```python
@abstractmethod
def open(self, run_dir: Path) -> "JudgementWriter":
    """Initialize for streaming."""
    pass

@abstractmethod
def write_one(self, judgement: LLMJudgement) -> None:
    """Write single judgement immediately (called in context)."""
    pass

@abstractmethod
def close(self) -> None:
    """Clean up (called automatically by context manager)."""
    pass
```

### Adding a New Prompt Format

1. Create builder in `adapters/prompts/` implementing `PromptBuilder` port
2. Create parser in `adapters/parsers/` implementing `ResponseParser` port
3. Update factories
4. Add config to `configs/prompts/` bundling builder + parser

## Testing Strategy

**Unit tests** (domain layer):
- Mock all ports (simple test doubles)
- Test domain logic in complete isolation
- Fast, deterministic

**Integration tests** (adapters):
- Test concrete adapters with real file I/O or API mocking
- Verify port contracts are satisfied

**CLI integration tests:**
- End-to-end with test fixtures
- Verify CLI flags → orchestrator → domain → adapters → output files

## Key Takeaways

1. **Hexagonal architecture** enables testable, swappable components
2. **Streaming with immediate persistence** provides fault tolerance and live progress
3. **Context managers** handle resource lifecycle cleanly (files, connections)
4. **Simple interfaces** (avoid iterators/async unless needed) reduce complexity
5. **Factory pattern** enables configuration-driven behavior
6. **Explicit configuration** makes system transparent and predictable
7. **Domain service** coordinates pure business logic using only port abstractions

These patterns ensure the CLI is maintainable, testable, and extensible for research experimentation.
