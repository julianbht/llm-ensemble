"""Aggregation strategy parameter type for aggregate CLI."""

from __future__ import annotations

import click

from llm_ensemble.aggregate.adapters.driven.aggregation_strategy_factory import AggregationStrategyBuilder


class AggregationStrategyParamType(click.ParamType):
    """Click parameter type for aggregation strategy selection.
    
    Reads available strategies from AggregationStrategyBuilder.
    """
    
    name = "AGGREGATION_STRATEGY"
    
    def convert(self, value, param, ctx):  # type: ignore[override]
        if value in (None, ""):
            self.fail(
                "Aggregation strategy name is required. Use --aggregation-strategy <name>.\n"
                "Run 'aggregate --help' to see available strategies.",
                param,
                ctx,
            )
        
        if not AggregationStrategyBuilder.has_strategy(value):
            available = AggregationStrategyBuilder.list_available()
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
        available = AggregationStrategyBuilder.list_available()
        return [
            click.shell_completion.CompletionItem(strategy)
            for strategy in available
            if strategy.startswith(incomplete)
        ]
