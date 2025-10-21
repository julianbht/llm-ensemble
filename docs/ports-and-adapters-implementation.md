# Ports & Adapters Architecture Implementation

This document summarizes the ports and adapters architecture implementation for prompt builders and response parsers in the infer CLI.

## What Changed

Previously, prompt builders and parsers were **utility functions** loaded dynamically via `importlib`. They were not part of the formal ports & adapters architecture.

Now, they are **proper port abstractions** with concrete adapter implementations, consistent with the rest of the infer CLI architecture.

## New Ports

### 1. PromptBuilder Port (`ports/prompt_builder.py`)

Abstract interface for building prompts from templates and judging examples.

```python
class PromptBuilder(ABC):
    @abstractmethod
    def build(self, template: Template, example: dict) -> str:
        """Build a prompt from template and judging example."""
        pass
```

### 2. ResponseParser Port (`ports/response_parser.py`)

Abstract interface for parsing LLM responses to extract relevance labels.

```python
class ResponseParser(ABC):
    @abstractmethod
    def parse(self, raw_text: str) -> tuple[Optional[int], list[str]]:
        """Parse LLM response to extract relevance label and warnings."""
        pass
```

## Concrete Adapters

### 1. ThomasPromptBuilder (`prompt_builders/thomas.py`)

Implements `PromptBuilder` for the Thomas et al. prompt format.

**Before:**
```python
def build(template, example):
    return template.render(...)
```

**After:**
```python
class ThomasPromptBuilder(PromptBuilder):
    def build(self, template, example):
        # Validate required fields
        if "query_text" not in example:
            raise ValueError("Example missing required field: query_text")
        
        return template.render(
            query=example["query_text"],
            page_text=example["doc"],
        )
```

### 2. ThomasResponseParser (`response_parsers/thomas.py`)

Implements `ResponseParser` for Thomas et al. JSON output format.

**Before:**
```python
def parse(raw_text):
    warnings = []
    # Extract JSON and parse
    return (label, warnings)
```

**After:**
```python
class ThomasResponseParser(ResponseParser):
    def parse(self, raw_text):
        warnings = []
        # Extract JSON and parse
        return (label, warnings)
```

## Factory Functions

Both factories now return concrete adapter instances rather than modules:

### load_builder() (`prompt_builders/__init__.py`)

```python
def load_builder(builder_name: str) -> PromptBuilder:
    """Load a prompt builder adapter by name.
    
    Returns:
        An instance of the PromptBuilder implementation
    """
    # Dynamically load module
    # Instantiate class (e.g., "thomas" → ThomasPromptBuilder())
    return builder_instance
```

### load_parser() (`response_parsers/__init__.py`)

```python
def load_parser(parser_name: str) -> ResponseParser:
    """Load a response parser adapter by name.
    
    Returns:
        An instance of the ResponseParser implementation
    """
    # Dynamically load module
    # Instantiate class (e.g., "thomas" → ThomasResponseParser())
    return parser_instance
```

## Benefits

### 1. Architectural Consistency

All I/O boundaries are now defined by ports:
- ✅ `LLMProvider` — LLM inference
- ✅ `ExampleReader` — Input reading
- ✅ `JudgementWriter` — Output writing
- ✅ `PromptBuilder` — Prompt construction (NEW)
- ✅ `ResponseParser` — Response parsing (NEW)

### 2. Explicit Dependencies

Providers and domain services can declare dependencies on port abstractions:

```python
class OpenRouterAdapter(LLMProvider):
    def infer(self, examples, model_config, ...):
        builder: PromptBuilder = load_builder(...)
        parser: ResponseParser = load_parser(...)
        
        for example in examples:
            prompt = builder.build(template, example)
            raw_text = api_call(prompt)
            label, warnings = parser.parse(raw_text)
```

### 3. Type Safety

Functions now have proper type hints:
```python
# Before
builder = load_builder("thomas")  # Returns module
prompt = builder.build(template, example)

# After
builder = load_builder("thomas")  # Returns PromptBuilder
prompt = builder.build(template, example)  # IDE knows method signature
```

### 4. Testability

Easy to create mock implementations for testing:

```python
class MockPromptBuilder(PromptBuilder):
    def build(self, template, example):
        return "mock prompt"

class MockResponseParser(ResponseParser):
    def parse(self, raw_text):
        return (2, [])  # Always return label=2
```

### 5. Extensibility

Clear contract for adding new prompt formats:

1. Implement `PromptBuilder` and `ResponseParser` ports
2. Add to `prompt_builders/` and `response_parsers/` directories
3. Register in prompt config YAML
4. No changes needed to domain logic or providers

## Configuration Coupling

Prompt configs bundle builders and parsers together (similar to I/O configs):

```yaml
# configs/prompts/thomas-et-al-prompt.yaml
name: thomas-et-al-prompt
prompt_template: thomas-et-al-prompt  # References .jinja file
prompt_builder: thomas                 # References prompt_builders/thomas.py
response_parser: thomas                # References response_parsers/thomas.py
```

This makes sense because:
- A prompt format defines **both** how to build prompts AND how to parse responses
- They're tightly coupled — changing the prompt format requires changing the parser
- Similar to I/O configs that bundle reader + writer for a format

## Migration Notes

### Breaking Changes

None! The factory functions maintain backward compatibility:

**Before:**
```python
builder = load_builder("thomas")
prompt = builder.build(template, example)
```

**After:**
```python
builder = load_builder("thomas")  # Now returns instance, not module
prompt = builder.build(template, example)  # Same API
```

### For Adding New Formats

**Before:**
Create a module with `build()` and `parse()` functions.

**After:**
Create classes implementing `PromptBuilder` and `ResponseParser` ports:

```python
# prompt_builders/my_format.py
class MyFormatPromptBuilder(PromptBuilder):
    def build(self, template, example):
        return template.render(...)

# response_parsers/my_format.py
class MyFormatResponseParser(ResponseParser):
    def parse(self, raw_text):
        return (label, warnings)
```

Class names must follow convention: `{CamelCaseName}PromptBuilder` / `{CamelCaseName}ResponseParser`

## Verification

All components verified working:

```bash
$ python -c "from llm_ensemble.infer.prompt_builders import load_builder; ..."
✓ Factories work
✓ Builder type: ThomasPromptBuilder
✓ Parser type: ThomasResponseParser
✓ Builder is PromptBuilder: True
✓ Parser is ResponseParser: True
```

## Next Steps

1. ✅ Ports defined and documented
2. ✅ Existing adapters refactored to implement ports
3. ✅ Factory functions updated
4. ✅ README updated with architecture details
5. ⏭️  Update tests to use mock implementations (deferred per user request)
6. ⏭️  Consider adding validation in factories (type checking, ABC compliance)

## Summary

The infer CLI now has a complete and consistent ports & adapters architecture. All I/O boundaries are defined by abstract port interfaces, enabling dependency inversion, testability, and clean separation between domain logic and infrastructure concerns.
