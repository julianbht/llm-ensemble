#!/usr/bin/env python3
"""Analyze maximum token counts in dataset for a given prompt template.

This script:
1. Loads samples from the database (ingest schema)
2. Renders prompts using a specified prompt builder
3. Counts tokens for each prompt using tiktoken
4. Reports statistics and identifies longest prompts

Usage:
    python scripts/analyze_token_counts.py --ingest-run <run_name> --prompt thomas-advanced-trec
    python scripts/analyze_token_counts.py --ingest-run <run_name> --prompt thomas-advanced-trec --limit 100
"""

import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load runtime configuration (DATABASE_URL, API keys, etc.)
from llm_ensemble.libs.runtime.env import load_runtime_config
load_runtime_config()

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import get_session
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    IngestRunORM,
    NormalizedDatasetJudgingSampleORM,
    JudgingSampleORM,
)
from llm_ensemble.ingest.domain.entities.dataset_sample import NormalizedDatasetJudgingSample
from llm_ensemble.ingest.domain.entities.judging_sample import JudgingSample
from llm_ensemble.ingest.domain.entities.query import Query
from llm_ensemble.ingest.domain.entities.document import Document

app = typer.Typer(
    add_completion=True,
    help="Analyze token counts for prompts in an ingested dataset",
    pretty_exceptions_enable=False,
)


def load_prompt_builder(prompt_name: str):
    """Load a prompt builder by name using the factory.

    Args:
        prompt_name: Prompt template name (e.g., 'thomas-advanced-trec')

    Returns:
        Prompt builder instance

    Raises:
        ValueError: If prompt builder not found
    """
    from llm_ensemble.infer.adapters.driven.prompt_factory import PromptAdapterFactory

    return PromptAdapterFactory.create(prompt_name)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to tokenize
        model: Model name for tokenizer (default: gpt-4)

    Returns:
        Number of tokens
    """
    try:
        import tiktoken
    except ImportError:
        print("Error: tiktoken not installed. Install with: pip install tiktoken", file=sys.stderr)
        raise typer.Exit(code=1)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def load_samples_from_db(ingest_run_name: str, limit: Optional[int] = None) -> list[NormalizedDatasetJudgingSample]:
    """Load samples from database for a given ingest run.

    Args:
        ingest_run_name: Name of the ingest run
        limit: Optional limit on number of samples

    Returns:
        List of NormalizedDatasetJudgingSample domain objects
    """
    engine = get_engine()
    session = get_session(engine)

    try:
        # Find ingest run by name
        ingest_run = session.query(IngestRunORM).filter_by(run_name=ingest_run_name).first()

        if not ingest_run:
            print(f"Error: Ingest run not found: {ingest_run_name}", file=sys.stderr)
            print(f"\nAvailable ingest runs:", file=sys.stderr)
            runs = session.query(IngestRunORM).order_by(IngestRunORM.created_at.desc()).limit(20).all()
            for run in runs:
                print(f"  - {run.run_name}", file=sys.stderr)
            raise typer.Exit(code=1)

        # Load samples with eager loading of relationships
        query = (
            select(NormalizedDatasetJudgingSampleORM)
            .where(NormalizedDatasetJudgingSampleORM.normalized_dataset_id == ingest_run.normalized_dataset_id)
            .options(
                joinedload(NormalizedDatasetJudgingSampleORM.judging_sample)
                .joinedload(JudgingSampleORM.query)
            )
            .options(
                joinedload(NormalizedDatasetJudgingSampleORM.judging_sample)
                .joinedload(JudgingSampleORM.document)
            )
            .order_by(NormalizedDatasetJudgingSampleORM.sequence_number)
        )

        if limit:
            query = query.limit(limit)

        sample_orms = session.execute(query).scalars().all()

        # Convert ORMs to domain objects
        samples = []
        for sample_orm in sample_orms:
            judging_sample_orm = sample_orm.judging_sample

            sample = NormalizedDatasetJudgingSample(
                id=sample_orm.id,
                normalized_dataset_id=sample_orm.normalized_dataset_id,
                sequence_number=sample_orm.sequence_number,
                judging_sample=JudgingSample(
                    id=judging_sample_orm.id,
                    query=Query(
                        id=judging_sample_orm.query.id,
                        content_hash=judging_sample_orm.query.content_hash,
                        query_text=judging_sample_orm.query.query_text,
                    ),
                    document=Document(
                        id=judging_sample_orm.document.id,
                        content_hash=judging_sample_orm.document.content_hash,
                        doc_text=judging_sample_orm.document.doc_text,
                    ),
                    gold_score=judging_sample_orm.gold_score,
                ),
            )
            samples.append(sample)

        return samples

    finally:
        session.close()


@app.command()
def analyze(
    ingest_run: Annotated[
        str,
        typer.Option(
            "--ingest-run",
            help="Name of the ingest run to analyze (from database)",
        ),
    ],
    prompt: Annotated[
        str,
        typer.Option(
            "--prompt",
            help="Prompt template name (e.g., 'thomas-advanced-trec', 'thomas-et-al-prompt')",
        ),
    ],
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit",
            help="Limit number of samples to analyze (for quick testing)",
        ),
    ] = None,
    show_top: Annotated[
        int,
        typer.Option(
            "--show-top",
            help="Show details for N longest prompts",
        ),
    ] = 5,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Model name for tokenizer (default: gpt-4)",
        ),
    ] = "gpt-4",
) -> None:
    """Analyze token counts for prompts in an ingested dataset.

    Examples:

        # Analyze all samples from an ingest run
        python scripts/analyze_token_counts.py analyze --ingest-run test/llm_judge_ingest_2024-01-09 --prompt thomas-advanced-trec

        # Quick test with first 100 samples
        python scripts/analyze_token_counts.py analyze --ingest-run test/llm_judge_ingest_2024-01-09 --prompt thomas-advanced-trec --limit 100

        # Show top 10 longest prompts
        python scripts/analyze_token_counts.py analyze --ingest-run test/llm_judge_ingest_2024-01-09 --prompt thomas-advanced-trec --show-top 10
    """
    print("Token Count Analysis")
    print("=" * 60)
    print(f"Ingest run:     {ingest_run}")
    print(f"Prompt:         {prompt}")
    print(f"Tokenizer:      {model}")
    print(f"Sample limit:   {limit or 'None (all samples)'}")
    print()

    # Load prompt builder
    try:
        builder = load_prompt_builder(prompt)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        raise typer.Exit(code=1)

    # Get template info
    template_text = builder.get_template_text()
    template_tokens = count_tokens(template_text, model)
    print(f"Template base tokens: {template_tokens:,}")
    print()

    # Load samples from database
    print("Loading samples from database...")
    samples = load_samples_from_db(ingest_run, limit)
    print(f"Loaded {len(samples):,} samples")
    print()

    # Analyze token counts
    print("Analyzing token counts...")
    token_counts = []
    longest_prompts = []  # [(token_count, query_text, doc_text, full_prompt), ...]

    for sample in samples:
        # Build prompt
        prompt_text = builder.build_prompt(sample)

        # Count tokens
        tokens = count_tokens(prompt_text, model)
        token_counts.append(tokens)

        # Track longest prompts
        query_text = sample.judging_sample.query.query_text
        doc_text = sample.judging_sample.document.doc_text
        longest_prompts.append((tokens, query_text, doc_text, prompt_text))

    # Sort by token count
    longest_prompts.sort(reverse=True, key=lambda x: x[0])

    # Statistics
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)
    avg_tokens = sum(token_counts) / len(token_counts)
    median_tokens = sorted(token_counts)[len(token_counts) // 2]

    # Percentiles
    p95_idx = int(len(token_counts) * 0.95)
    p99_idx = int(len(token_counts) * 0.99)
    sorted_counts = sorted(token_counts)
    p95_tokens = sorted_counts[p95_idx]
    p99_tokens = sorted_counts[p99_idx]

    print("=" * 60)
    print("STATISTICS")
    print("=" * 60)
    print(f"Total samples:    {len(samples):,}")
    print()
    print(f"Min tokens:       {min_tokens:,}")
    print(f"Max tokens:       {max_tokens:,}")
    print(f"Average tokens:   {avg_tokens:,.1f}")
    print(f"Median tokens:    {median_tokens:,}")
    print()
    print(f"95th percentile:  {p95_tokens:,}")
    print(f"99th percentile:  {p99_tokens:,}")
    print()

    # Context window warnings
    context_limits = {
        "gemini-flash-1.5": 1_000_000,
        "gemini-pro-1.5": 2_000_000,
        "gpt-4": 8_192,
        "gpt-4-32k": 32_768,
        "gpt-4-turbo": 128_000,
        "claude-3-opus": 200_000,
        "claude-3.5-sonnet": 200_000,
    }

    print("CONTEXT WINDOW COMPATIBILITY")
    print("=" * 60)
    for model_name, context_limit in context_limits.items():
        exceeds_count = sum(1 for t in token_counts if t > context_limit)
        if exceeds_count > 0:
            print(f"❌ {model_name:25s} ({context_limit:>9,} tokens): {exceeds_count:,} samples exceed limit ({exceeds_count/len(samples)*100:.1f}%)")
        else:
            print(f"✓  {model_name:25s} ({context_limit:>9,} tokens): All samples fit")
    print()

    # Show longest prompts
    print(f"TOP {show_top} LONGEST PROMPTS")
    print("=" * 60)
    for i, (tokens, query, doc, full_prompt) in enumerate(longest_prompts[:show_top], 1):
        query_tokens = count_tokens(query, model)
        doc_tokens = count_tokens(doc, model)

        print(f"\n{i}. Total tokens: {tokens:,}")
        print(f"   Query tokens:    {query_tokens:,}")
        print(f"   Document tokens: {doc_tokens:,}")
        print(f"   Template tokens: {template_tokens:,}")
        print(f"   Query preview:   {query[:100]}..." if len(query) > 100 else f"   Query:           {query}")
        print(f"   Document chars:  {len(doc):,} chars")
        print(f"   Document preview: {doc[:150]}..." if len(doc) > 150 else f"   Document:        {doc}")

    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print(f"• Use models with context windows >= {max_tokens:,} tokens")
    print(f"• Consider filtering/truncating documents > {p95_tokens:,} tokens (95th percentile)")
    print(f"• For max compatibility, implement document truncation to stay under 32k tokens")
    print()


if __name__ == "__main__":
    app()
