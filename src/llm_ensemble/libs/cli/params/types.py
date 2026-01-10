"""Param type helpers for CLI options."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

import click

from llm_ensemble.libs.cli.error_messages import (
    format_invalid_config_error,
    format_invalid_io_config_error,
    format_missing_config_error,
    format_missing_io_config_error,
)
from llm_ensemble.libs.runtime.path_manager import PathManager

MissingFactory = Callable[[Path, List[str], str], str]
InvalidFactory = Callable[[str, Path, List[str], str], str]


def list_available_configs(config_dir: Path) -> list[str]:
    """List available YAML configs in directory."""
    if not config_dir.exists():
        return []
    return sorted([p.stem for p in config_dir.glob("*.yaml")])


class ConfigParamType(click.ParamType):
    """Base Click parameter type for config selectors."""

    name = "TEXT"

    def __init__(
        self,
        *,
        param_name: str,
        config_type_label: str,
        config_dir_provider: Callable[[], Path],
        example_fallback: str,
        missing_factory: MissingFactory | None = None,
        invalid_factory: InvalidFactory | None = None,
        allow_empty: bool = False,
    ) -> None:
        self.param_name = param_name
        self.config_type_label = config_type_label
        self._config_dir_provider = config_dir_provider
        self.example_fallback = example_fallback
        self._missing_factory = missing_factory
        self._invalid_factory = invalid_factory
        self.allow_empty = allow_empty

    def _config_dir(self) -> Path:
        return self._config_dir_provider()

    def _available(self, config_dir: Path) -> List[str]:
        return list_available_configs(config_dir)

    def _example(self, available: List[str]) -> str:
        return available[0] if available else self.example_fallback

    def _format_missing(self, config_dir: Path, available: List[str]) -> str:
        example = self._example(available)
        if self._missing_factory:
            return self._missing_factory(config_dir, available, example)
        return format_missing_config_error(
            param_name=self.param_name,
            config_type=self.config_type_label,
            config_dir=config_dir,
            available=available,
            example=example,
        )

    def _format_invalid(
        self, invalid_value: str, config_dir: Path, available: List[str]
    ) -> str:
        example = self._example(available)
        if self._invalid_factory:
            return self._invalid_factory(invalid_value, config_dir, available, example)
        return format_invalid_config_error(
            param_name=self.param_name,
            config_type=self.config_type_label,
            config_dir=config_dir,
            available=available,
            example=example,
            invalid_value=invalid_value,
        )

    def get_missing_message(self, param, ctx):  # type: ignore[override]
        config_dir = self._config_dir()
        available = self._available(config_dir)
        return self._format_missing(config_dir, available)

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            if self.allow_empty:
                return None
            self.fail(self.get_missing_message(param, ctx), param, ctx)

        config_dir = self._config_dir()
        config_path = config_dir / f"{value}.yaml"
        if not config_path.exists():
            available = self._available(config_dir)
            self.fail(
                self._format_invalid(value, config_dir, available),
                param,
                ctx,
            )
        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available configs."""
        config_dir = self._config_dir()
        available = self._available(config_dir)
        return [
            click.shell_completion.CompletionItem(cfg)
            for cfg in available
            if cfg.startswith(incomplete)
        ]


class IngestIOConfigParamType(click.ParamType):
    """Click parameter type for ingest IO format selection."""

    name = "IO_FORMAT"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.ingest.adapters.driven.io_factory import IOAdapterFactory
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format is required. Use --io-cfg <format>.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.ingest.adapters.driven.io_factory import IOAdapterFactory
        if not IOAdapterFactory.has_format(value):
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format '{value}' not found.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available IO formats."""
        from llm_ensemble.ingest.adapters.driven.io_factory import IOAdapterFactory
        available = IOAdapterFactory.list_available()
        return [
            click.shell_completion.CompletionItem(fmt)
            for fmt in available
            if fmt.startswith(incomplete)
        ]


class InferIOConfigParamType(click.ParamType):
    """Click parameter type for infer IO format selection."""

    name = "IO_FORMAT"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.infer.adapters.driven.io_factory import IOAdapterFactory
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format is required. Use --io-cfg <format>.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.infer.adapters.driven.io_factory import IOAdapterFactory
        if not IOAdapterFactory.has_format(value):
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format '{value}' not found.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available IO formats."""
        from llm_ensemble.infer.adapters.driven.io_factory import IOAdapterFactory
        available = IOAdapterFactory.list_available()
        return [
            click.shell_completion.CompletionItem(fmt)
            for fmt in available
            if fmt.startswith(incomplete)
        ]


class AggregateIOConfigParamType(click.ParamType):
    """Click parameter type for aggregate IO format selection."""

    name = "IO_FORMAT"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.aggregate.adapters.driven.io_factory import IOAdapterFactory
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format is required. Use --io-cfg <format>.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.aggregate.adapters.driven.io_factory import IOAdapterFactory
        if not IOAdapterFactory.has_format(value):
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format '{value}' not found.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available IO formats."""
        from llm_ensemble.aggregate.adapters.driven.io_factory import IOAdapterFactory
        available = IOAdapterFactory.list_available()
        return [
            click.shell_completion.CompletionItem(fmt)
            for fmt in available
            if fmt.startswith(incomplete)
        ]


class EvaluateIOConfigParamType(click.ParamType):
    """Click parameter type for evaluate IO format selection."""

    name = "IO_FORMAT"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format is required. Use --io-cfg <format>.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory
        if not IOAdapterFactory.has_format(value):
            available = IOAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"IO format '{value}' not found.\n"
                f"Available formats: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available IO formats."""
        from llm_ensemble.evaluate.adapters.driven.io_factory import IOAdapterFactory
        available = IOAdapterFactory.list_available()
        return [
            click.shell_completion.CompletionItem(fmt)
            for fmt in available
            if fmt.startswith(incomplete)
        ]


class ModelConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--model-cfg",
            config_type_label="model",
            config_dir_provider=PathManager.get_model_configs_dir,
            example_fallback="gpt-oss-20b-free",
        )


class LogConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--log-cfg",
            config_type_label="logging",
            config_dir_provider=lambda: PathManager.get_configs_dir() / "logging",
            example_fallback="standard",
            allow_empty=True,
        )


class RetryConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--retry-cfg",
            config_type_label="retry",
            config_dir_provider=PathManager.get_retries_dir,
            example_fallback="standard",
        )


class PromptTemplateParamType(click.ParamType):
    """Click parameter type for prompt template selection.

    Reads available templates from PromptTemplateFactory.
    Each template bundles a prompt builder and response parser together.
    """

    name = "TEMPLATE"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.infer.adapters.driven.prompt_factory import PromptAdapterFactory
            available = PromptAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"Template name is required. Use --prompt-template <name>.\n"
                f"Available templates: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.infer.adapters.driven.prompt_factory import PromptAdapterFactory
        if not PromptAdapterFactory.has_prompt(value):
            available = PromptAdapterFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"Template '{value}' not found.\n"
                f"Available templates: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available templates."""
        from llm_ensemble.infer.adapters.driven.prompt_factory import PromptAdapterFactory
        available = PromptAdapterFactory.list_available()
        return [
            click.shell_completion.CompletionItem(template)
            for template in available
            if template.startswith(incomplete)
        ]


class ProviderParamType(click.ParamType):
    """Click parameter type for provider adapter selection.

    Reads available providers from ProviderAdapterBuilder.
    """

    name = "PROVIDER"

    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            from llm_ensemble.infer.adapters.driven.provider_factory import ProviderFactory
            available = ProviderFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"Provider name is required. Use --provider <name>.\n"
                f"Available providers: {available_str}",
                param,
                ctx,
            )

        from llm_ensemble.infer.adapters.driven.provider_factory import ProviderFactory
        if not ProviderFactory.has_provider(value):
            available = ProviderFactory.list_available()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"Provider '{value}' not found.\n"
                f"Available providers: {available_str}",
                param,
                ctx,
            )

        return value

    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available providers."""
        from llm_ensemble.infer.adapters.driven.provider_factory import ProviderFactory
        available = ProviderFactory.list_available()
        return [
            click.shell_completion.CompletionItem(provider)
            for provider in available
            if provider.startswith(incomplete)
        ]
