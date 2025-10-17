"""Prompts feature for building prompts from templates.

The prompts module contains builder modules that know how to render
templates with specific data. Each builder is in prompts/builders/.
"""

from llm_ensemble.infer.prompts.builders import load_builder

__all__ = ["load_builder"]
