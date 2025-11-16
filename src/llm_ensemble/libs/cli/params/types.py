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
    ) -> None:
        self.param_name = param_name
        self.config_type_label = config_type_label
        self._config_dir_provider = config_dir_provider
        self.example_fallback = example_fallback
        self._missing_factory = missing_factory
        self._invalid_factory = invalid_factory

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
        if not value:
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


class IOConfigParamType(ConfigParamType):
    """Click parameter type that validates CLI-specific I/O configs."""

    def __init__(self, cli_name: str) -> None:
        super().__init__(
            param_name="--io-cfg",
            config_type_label=f"{cli_name} I/O",
            config_dir_provider=lambda: PathManager.get_io_configs_dir(cli_name),
            example_fallback="json",
            missing_factory=lambda config_dir, available, example: format_missing_io_config_error(
                cli_name, config_dir=config_dir, available=available
            ),
            invalid_factory=lambda invalid_value, config_dir, available, example: format_invalid_io_config_error(
                cli_name=cli_name,
                invalid_value=invalid_value,
                config_dir=config_dir,
                available=available,
            ),
        )


class ModelConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--model-cfg",
            config_type_label="model",
            config_dir_provider=PathManager.get_model_configs_dir,
            example_fallback="gpt-oss-20b-free",
        )


class PromptConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--prompt-cfg",
            config_type_label="prompt",
            config_dir_provider=PathManager.get_prompts_dir,
            example_fallback="thomas-simple",
        )


class EnsembleConfigParamType(ConfigParamType):
    def __init__(self) -> None:
        super().__init__(
            param_name="--ensemble-cfg",
            config_type_label="ensemble",
            config_dir_provider=PathManager.get_ensembles_dir,
            example_fallback="weighted_majority_v1",
        )
