"""Driven ports (secondary/repository ports).

Driven ports are infrastructure interfaces that the application USES (calls out to).
These are defined BY the application, implemented BY driven adapters (infrastructure).

The application depends on these abstractions, not on concrete implementations.
This enables dependency inversion and testability.
"""
