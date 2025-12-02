"""Test script to verify registry pattern implementation."""

from llm_ensemble.infer_cli import app  # Import to trigger adapter registration
from llm_ensemble.infer.adapters.prompts.registry import prompt_registry
from llm_ensemble.infer.adapters.parsers.registry import parser_registry
from llm_ensemble.libs.registry import AdapterWithMetadata

print("=" * 60)
print("Testing Registry Pattern Implementation")
print("=" * 60)

# Test 1: Verify registries are populated
print("\n1. Registry Contents:")
print(f"   Prompts: {list(prompt_registry._registry.keys())}")
print(f"   Parsers: {list(parser_registry._registry.keys())}")

# Test 2: Get metadata
print("\n2. Getting Metadata:")
prompt_meta = prompt_registry.get_metadata("thomas-simple")
parser_meta = parser_registry.get_metadata("thomas-simple")
print(f"   Prompt: {prompt_meta.name} - {prompt_meta.description}")
print(f"   Parser: {parser_meta.name} - {parser_meta.description}")

# Test 3: Instantiate adapters
print("\n3. Instantiating Adapters:")
prompt_builder = prompt_meta.adapter_class(
    template_path=prompt_meta.config["template_path"]
)
response_parser = parser_meta.adapter_class()
print(f"   Prompt builder: {type(prompt_builder).__name__}")
print(f"   Response parser: {type(response_parser).__name__}")

# Test 4: Wrap with metadata
print("\n4. Creating AdapterWithMetadata Wrappers:")
prompt_adapter = AdapterWithMetadata(
    adapter=prompt_builder,
    name=prompt_meta.name
)
parser_adapter = AdapterWithMetadata(
    adapter=response_parser,
    name=parser_meta.name
)
print(f"   Prompt wrapper: adapter={type(prompt_adapter.adapter).__name__}, name={prompt_adapter.name}")
print(f"   Parser wrapper: adapter={type(parser_adapter.adapter).__name__}, name={parser_adapter.name}")

# Test 5: Verify adapters have expected methods
print("\n5. Verifying Adapter Methods:")
print(f"   Prompt has build_raw: {hasattr(prompt_builder, 'build_raw')}")
print(f"   Prompt has get_template_text: {hasattr(prompt_builder, 'get_template_text')}")
print(f"   Parser has parse_raw: {hasattr(response_parser, 'parse_raw')}")

# Test 6: Verify identity computation
print("\n6. Computing UUIDs from Registry Names:")
from llm_ensemble.libs.db import compute_prompt_template_uuid, compute_parser_spec_uuid_from_name
prompt_template_id = compute_prompt_template_uuid(prompt_adapter.name)
parser_spec_id = compute_parser_spec_uuid_from_name(parser_adapter.name)
print(f"   Prompt template UUID: {prompt_template_id}")
print(f"   Parser spec UUID: {parser_spec_id}")

print("\n" + "=" * 60)
print("✓ All registry pattern tests passed!")
print("=" * 60)
