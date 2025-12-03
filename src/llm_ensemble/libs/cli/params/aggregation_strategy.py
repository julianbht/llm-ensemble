"""Aggregation strategy parameter type for aggregate CLI."""

from __future__ import annotations

import click

from llm_ensemble.aggregate.registry import AggregationStrategyRegistry
from llm_ensemble.aggregate.strategy_registration import ensure_all_strategies_registered


class AggregationStrategyParamType(click.ParamType):
    """Click parameter type for aggregation strategy selection.
    
    Reads available strategies from AggregationStrategyRegistry for validation and completion.
    Ensures strategies are registered before use.
    """
    
    name = "AGGREGATION_STRATEGY"
    
    def __init__(self):
        super().__init__()
        # Ensure all strategies are registered when param type is created
        ensure_all_strategies_registered()
    
    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            self.fail(
                "Aggregation strategy name is required. Use --aggregation-strategy <name>.\n"
                "Run 'aggregate --help' to see available strategies.",
                param,
                ctx,
            )
        
        if not AggregationStrategyRegistry.has_strategy(value):
            available = AggregationStrategyRegistry.list_strategies()
            available_str = ", ".join(available) if available else "none"
            self.fail(
                f"Aggregation strategy '{value}' not found.\n"
                f"Available strategies: {available_str}",
                param,
                ctx,
            )
        
        return value
    
    def shell_complete(self, ctx, param, incomplete):  # type: ignore[override]
        """Provide shell completion for available strategies."""
        available = AggregationStrategyRegistry.list_strategies()
        return [
            click.shell_completion.CompletionItem(strategy)
            for strategy in available
            if strategy.startswith(incomplete)
        ]
