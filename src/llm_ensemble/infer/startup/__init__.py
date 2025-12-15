"""Startup layer for inference CLI.

Composition root - infrastructure wiring and dependency configuration.

This layer is responsible for:
- Loading configurations
- Setting up infrastructure (run directories, logging)
- Selecting and instantiating adapters based on configuration
- Building the application (use case) with injected dependencies
- Executing the use case
- Post-processing (manifests, summaries)

Not unit tested - tested via CLI integration tests.
"""
