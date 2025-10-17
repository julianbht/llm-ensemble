"""Prompts feature for template loading and rendering."""

from llm_ensemble.infer.prompts.templates import load_prompt_template
from llm_ensemble.infer.prompts.builder import build_instruction, build_instruction_from_judging_example

__all__ = [
    "load_prompt_template",
    "build_instruction",
    "build_instruction_from_judging_example",
]
