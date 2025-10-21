# Dependency Injection Refactoring

## Summary

Refactored the infer CLI to properly inject `PromptBuilder`, `ResponseParser`, and `Template` dependencies into LLM provider adapters, eliminating duplicate config loading and achieving consistent dependency injection across all adapters.

## Problem

**Before this refactoring:**
- The orchestrator loaded prompt configs but **didn't use them**
- Providers loaded the same configs **again** inside their `infer()` methods
- Builder and parser were created **inside each provider** (duplication)
- Inconsistent with how other adapters (reader, writer) were injected

```python
# ❌ OLD: Config loaded twice
orchestrator:
    prompt_config = load_prompt_config("thomas")  # Loaded but not used!
    provider = get_provider(model_config)
    service.run_inference(..., prompt_template_name="thomas")
        ↓
    provider.infer(..., prompt_template_name="thomas")
        # Loads config AGAIN inside provider!
        prompt_config = load_prompt_config("thomas")  # Duplicate!
        builder = load_builder(...)
        parser = load_parser(...)
```

## Solution

**After this refactoring:**
- Orchestrator loads configs **once** and creates all adapters
- Builder, parser, and template are **injected** into provider constructors
- Providers receive **all dependencies** at construction time
- Consistent dependency injection for all adapters

```python
# ✅ NEW: Config loaded once, dependencies injected
orchestrator:
    prompt_config = load_prompt_config("thomas")
    template = load_prompt_template(...)
    builder = load_builder(prompt_config.prompt_builder)  # Create once
    parser = load_parser(prompt_config.response_parser)   # Create once
    
    provider = get_provider(
        model_config,
        builder=builder,    # Inject
        parser=parser,      # Inject
        template=template   # Inject
    )
    
    service.run_inference(...)  # No prompt params needed!
        ↓
    provider.infer(examples, model_config)  # Uses injected dependencies
```

## Changes Made

### 1. Updated LLMProvider Port

**File:** `src/llm_ensemble/infer/ports/llm_provider.py`

**Before:**
```python
def infer(
    self,
    examples: Iterator[JudgingExample],
    model_config: ModelConfig,
    prompt_template_name: str,      # ← Removed
    prompts_dir: Optional[Path],    # ← Removed
) -> Iterator[ModelJudgement]:
```

**After:**
```python
def infer(
    self,
    examples: Iterator[JudgingExample],
    model_config: ModelConfig,
) -> Iterator[ModelJudgement]:
```

Dependencies now injected via constructor, not method parameters.

### 2. Updated Provider Adapters

**All providers** (OpenRouter, Ollama, HuggingFace) now:

**Constructor:**
```python
def __init__(
    self,
    builder: PromptBuilder,    # ← NEW: Injected
    parser: ResponseParser,     # ← NEW: Injected
    template: Template,         # ← NEW: Injected
    api_key: Optional[str] = None,
    timeout: int = 30,
):
    self.builder = builder      # Store dependencies
    self.parser = parser
    self.template = template
    # ... rest of initialization
```

**infer() method:**
```python
def infer(self, examples, model_config):
    for example in examples:
        # Use injected dependencies (no config loading!)
        prompt = self.builder.build(self.template, example.model_dump())
        response = api_call(prompt)
        label, warnings = self.parser.parse(response)
        yield ModelJudgement(...)
```

### 3. Updated Provider Factory

**File:** `src/llm_ensemble/infer/adapters/provider_factory.py`

**Before:**
```python
def get_provider(
    model_config: ModelConfig,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> LLMProvider:
    if model_config.provider == "openrouter":
        return OpenRouterAdapter(api_key=api_key, timeout=timeout)
```

**After:**
```python
def get_provider(
    model_config: ModelConfig,
    builder: PromptBuilder,    # ← NEW: Required
    parser: ResponseParser,     # ← NEW: Required
    template: Template,         # ← NEW: Required
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> LLMProvider:
    if model_config.provider == "openrouter":
        return OpenRouterAdapter(
            builder=builder,        # ← Pass through
            parser=parser,
            template=template,
            api_key=api_key,
            timeout=timeout,
        )
```

### 4. Updated InferenceService

**File:** `src/llm_ensemble/infer/domain/inference_service.py`

**Before:**
```python
def run_inference(
    self,
    input_path: Path,
    model_config: ModelConfig,
    prompt_template_name: str,      # ← Removed
    prompts_dir: Optional[Path],    # ← Removed
    limit: Optional[int] = None,
    on_judgement: Optional[Callable] = None,
) -> dict:
```

**After:**
```python
def run_inference(
    self,
    input_path: Path,
    model_config: ModelConfig,
    limit: Optional[int] = None,
    on_judgement: Optional[Callable] = None,
) -> dict:
```

Provider already configured, so no prompt parameters needed.

### 5. Updated Orchestrator

**File:** `src/llm_ensemble/infer/orchestrator.py`

**Key changes:**

```python
# Load prompt config and create adapters (NEW SECTION)
template = load_prompt_template(prompt_config.prompt_template, prompts_dir)
builder = load_builder(prompt_config.prompt_builder)
parser = load_parser(prompt_config.response_parser)

logger.info(
    "Loaded prompt components",
    template=prompt_config.prompt_template,
    builder=prompt_config.prompt_builder,
    parser=prompt_config.response_parser,
)

# Create provider with injected dependencies
provider = get_provider(
    model_config,
    builder=builder,    # ← NEW: Inject
    parser=parser,      # ← NEW: Inject
    template=template,  # ← NEW: Inject
)

# Run inference (no prompt params needed)
stats = service.run_inference(
    input_path=input_file,
    model_config=model_config,
    limit=limit,              # Removed: prompt_template_name
    on_judgement=log_judgement,  # Removed: prompts_dir
)
```

## Benefits

### 1. Single Responsibility Principle
- **Orchestrator**: Wires dependencies (loads configs, creates adapters)
- **Providers**: Handle API communication (use injected dependencies)
- **Domain Service**: Pure business logic (no config knowledge)

### 2. No Duplication
- Config files read **once** (not once per provider)
- Builder/parser created **once** (not in every provider)
- Template loaded **once** (not repeatedly)

### 3. Explicit Dependencies
All dependencies visible in constructors:
```python
provider = OpenRouterAdapter(
    builder=builder,    # Explicit
    parser=parser,      # Explicit
    template=template,  # Explicit
    api_key=api_key
)
```

### 4. Easier Testing
```python
# Mock all dependencies for testing
mock_builder = MockPromptBuilder()
mock_parser = MockResponseParser()
mock_template = Mock()

provider = OpenRouterAdapter(
    builder=mock_builder,
    parser=mock_parser,
    template=mock_template,
    api_key="test"
)

# Test provider without any config files!
```

### 5. Performance
- No redundant file I/O (config loaded once, not per provider)
- No redundant imports (builder/parser instantiated once)

### 6. Architectural Consistency
**Before:** Inconsistent injection
```
✓ ExampleReader injected by orchestrator
✓ JudgementWriter injected by orchestrator
✗ PromptBuilder created inside provider
✗ ResponseParser created inside provider
```

**After:** Consistent injection
```
✓ ExampleReader injected by orchestrator
✓ JudgementWriter injected by orchestrator
✓ PromptBuilder injected by orchestrator
✓ ResponseParser injected by orchestrator
✓ Template injected by orchestrator
```

## Migration Notes

### Breaking Changes

**LLMProvider port signature changed:**
- Removed `prompt_template_name` parameter
- Removed `prompts_dir` parameter
- Dependencies now passed via constructor

**Provider constructors changed:**
- Added `builder: PromptBuilder` parameter
- Added `parser: ResponseParser` parameter
- Added `template: Template` parameter

**InferenceService.run_inference() changed:**
- Removed `prompt_template_name` parameter
- Removed `prompts_dir` parameter

### Backward Compatibility

**Orchestrator API unchanged:**
- CLI still passes `prompt` parameter
- CLI still passes `prompts_dir` parameter
- Orchestrator handles the wiring internally

**Users don't see the change** - it's an internal refactoring.

## Files Modified

1. `src/llm_ensemble/infer/ports/llm_provider.py` - Updated port signature
2. `src/llm_ensemble/infer/adapters/providers/openrouter_adapter.py` - Dependency injection
3. `src/llm_ensemble/infer/adapters/providers/ollama_adapter.py` - Dependency injection
4. `src/llm_ensemble/infer/adapters/providers/huggingface_adapter.py` - Dependency injection
5. `src/llm_ensemble/infer/adapters/provider_factory.py` - Pass through dependencies
6. `src/llm_ensemble/infer/domain/inference_service.py` - Remove prompt params
7. `src/llm_ensemble/infer/orchestrator.py` - Create and inject dependencies

## Verification

All changes verified working:
```bash
✓ Port imports work
✓ Builder and parser loaded
✓ OpenRouterAdapter accepts injected dependencies
✓ Provider factory works with new signature
✓ Orchestrator signature correct
✓ Full dependency injection flow working
```

## Next Steps

This refactoring sets the foundation for:

1. **Template Method Pattern** - Could extract common provider logic to base class
2. **Multiple Prompt Formats** - Easy to switch by injecting different builder/parser
3. **Provider Testing** - Can now test with mocks without config files
4. **Better Error Handling** - Config errors caught at orchestrator level, not deep in providers

## Related Documentation

- [Ports & Adapters Implementation](./ports-and-adapters-implementation.md)
- [Architecture Diagram](./architecture-diagram.txt)
- [Infer CLI README](../src/llm_ensemble/infer/README.md)
