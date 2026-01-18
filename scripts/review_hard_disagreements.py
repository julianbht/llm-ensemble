#!/usr/bin/env python3
"""Interactive review tool for hard disagreements between LLM and human judgements.

Presents query-document pairs where the predicted label differs from human
annotation by 2+ levels (e.g., 0 vs 2, 1 vs 3). Supports blind review,
resume functionality, and go-back to fix mistakes.

Usage:
    # Review hard disagreements from ensemble majority vote average run
    python scripts/review_hard_disagreements.py --run-name "-ensemble-1-to-5-majority-vote-average" --run-type aggregate

    # Review hard disagreements from GPT reference run
    python scripts/review_hard_disagreements.py --run-name "gpt-5-1-all-samples-base-reference" --run-type infer

    # Resume a previous session
    python scripts/review_hard_disagreements.py --resume artifacts/review_sessions/session_2025-01-18_12-00-00.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.libs.schemas.relevance_score import RelevanceScore
from llm_ensemble.infer.adapters.driven.io.db.orms import (
    InferRunORM,
    LLMJudgementORM,
)
from llm_ensemble.aggregate.adapters.driven.io.orms import (
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    NormalizedDatasetJudgingSampleORM,
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
)
from sqlalchemy.orm import selectinload, joinedload


app = typer.Typer()
console = Console()

# Relevance scale descriptions for reference
RELEVANCE_LABELS = {
    0: "Irrelevant - Document is not relevant to the query",
    1: "Relevant - Document has some relevance",
    2: "Highly Relevant - Document is highly relevant",
    3: "Perfectly Relevant - Document perfectly addresses the query",
}


def load_hard_disagreements_from_infer(run_name: str) -> list[dict]:
    """Load hard disagreements from an infer run (single model)."""
    engine = get_engine()
    disagreements = []

    with session_context(engine) as session:
        # Find infer run
        infer_run = (
            session.query(InferRunORM)
            .filter_by(run_name=run_name)
            .options(joinedload(InferRunORM.infer_run_output))
            .one_or_none()
        )
        if not infer_run:
            raise ValueError(f"Infer run '{run_name}' not found")

        infer_run_output = infer_run.infer_run_output
        if not infer_run_output:
            raise ValueError(f"No output for infer run '{run_name}'")

        # Load all judgements
        judgements = (
            session.query(LLMJudgementORM)
            .filter_by(infer_run_output_id=infer_run_output.id)
            .options(joinedload(LLMJudgementORM.llm_score))
            .order_by(LLMJudgementORM.normalized_dataset_judging_sample_id)
            .all()
        )

        for j in judgements:
            # Get predicted label
            predicted = j.llm_score.label if j.llm_score else None
            if predicted is None:
                continue  # Skip parse failures

            # Load sample data
            sample = (
                session.query(
                    NormalizedDatasetJudgingSampleORM,
                    JudgingSampleORM,
                    QueryORM,
                    DocumentORM,
                )
                .join(
                    JudgingSampleORM,
                    NormalizedDatasetJudgingSampleORM.judging_sample_id
                    == JudgingSampleORM.id,
                )
                .join(QueryORM, JudgingSampleORM.query_id == QueryORM.id)
                .join(DocumentORM, JudgingSampleORM.document_id == DocumentORM.id)
                .filter(
                    NormalizedDatasetJudgingSampleORM.id
                    == j.normalized_dataset_judging_sample_id
                )
                .first()
            )

            if not sample:
                continue

            ds_orm, js_orm, query_orm, doc_orm = sample
            gold = js_orm.gold_score

            # Check for hard disagreement (difference >= 2)
            if abs(predicted - gold) >= 2:
                disagreements.append(
                    {
                        "sample_id": str(ds_orm.id),
                        "sequence_number": ds_orm.sequence_number,
                        "query_text": query_orm.query_text,
                        "doc_text": doc_orm.doc_text,
                        "gold_label": gold,
                        "predicted_label": predicted,
                        "disagreement_level": abs(predicted - gold),
                    }
                )

    return disagreements


def load_hard_disagreements_from_aggregate(run_name: str) -> list[dict]:
    """Load hard disagreements from an aggregate run (ensemble)."""
    engine = get_engine()
    disagreements = []

    with session_context(engine) as session:
        # Find aggregate run
        aggregate_run = (
            session.query(AggregateRunORM)
            .filter_by(run_name=run_name)
            .options(
                selectinload(AggregateRunORM.aggregated_dataset).selectinload(
                    AggregatedDatasetORM.aggregated_votes
                )
            )
            .first()
        )
        if not aggregate_run:
            raise ValueError(f"Aggregate run '{run_name}' not found")

        aggregated_dataset = aggregate_run.aggregated_dataset
        if not aggregated_dataset:
            raise ValueError(f"No dataset for aggregate run '{run_name}'")

        # Process each vote
        for vote in aggregated_dataset.aggregated_votes:
            predicted = vote.final_label
            if predicted is None:
                continue

            # Get the sample via the junction table
            # We need to find the sample_id from the aggregation_votes
            agg_vote = (
                session.query(AggregatedVoteORM)
                .filter_by(id=vote.id)
                .options(
                    selectinload(AggregatedVoteORM.aggregation_votes).joinedload(
                        "llm_judgement"
                    )
                )
                .first()
            )

            if not agg_vote or not agg_vote.aggregation_votes:
                continue

            # Get sample_id from first judgement (all should be same sample)
            sample_id = agg_vote.aggregation_votes[
                0
            ].llm_judgement.normalized_dataset_judging_sample_id

            # Load sample data
            sample = (
                session.query(
                    NormalizedDatasetJudgingSampleORM,
                    JudgingSampleORM,
                    QueryORM,
                    DocumentORM,
                )
                .join(
                    JudgingSampleORM,
                    NormalizedDatasetJudgingSampleORM.judging_sample_id
                    == JudgingSampleORM.id,
                )
                .join(QueryORM, JudgingSampleORM.query_id == QueryORM.id)
                .join(DocumentORM, JudgingSampleORM.document_id == DocumentORM.id)
                .filter(NormalizedDatasetJudgingSampleORM.id == sample_id)
                .first()
            )

            if not sample:
                continue

            ds_orm, js_orm, query_orm, doc_orm = sample
            gold = js_orm.gold_score

            # Check for hard disagreement (difference >= 2)
            if abs(predicted - gold) >= 2:
                disagreements.append(
                    {
                        "sample_id": str(ds_orm.id),
                        "sequence_number": ds_orm.sequence_number,
                        "query_text": query_orm.query_text,
                        "doc_text": doc_orm.doc_text,
                        "gold_label": gold,
                        "predicted_label": predicted,
                        "disagreement_level": abs(predicted - gold),
                    }
                )

    return disagreements


def create_session_file(
    run_name: str, run_type: str, disagreements: list[dict]
) -> Path:
    """Create a new review session file."""
    sessions_dir = Path(__file__).parent.parent / "artifacts" / "review_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_file = sessions_dir / f"session_{timestamp}.json"

    session_data = {
        "run_name": run_name,
        "run_type": run_type,
        "created_at": datetime.now().isoformat(),
        "total_disagreements": len(disagreements),
        "current_index": 0,
        "disagreements": disagreements,
        "reviews": [],  # Will contain {sample_id, user_label, notes, reviewed_at}
    }

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    return session_file


def load_session(session_file: Path) -> dict:
    """Load an existing session."""
    with open(session_file) as f:
        return json.load(f)


def save_session(session_file: Path, session_data: dict):
    """Save session state."""
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)


def display_sample_for_review(item: dict, index: int, total: int, blind: bool = True):
    """Display a query-document pair for review."""
    console.clear()

    # Header with progress
    console.print(
        Panel(
            f"[bold]Hard Disagreement Review[/bold] - Sample {index + 1} of {total}",
            style="blue",
        )
    )

    # Query panel
    console.print()
    console.print(Panel(item["query_text"], title="[bold cyan]Query[/bold cyan]"))

    # Document panel (may be long, so truncate for display)
    doc_text = item["doc_text"]
    if len(doc_text) > 2000:
        doc_display = doc_text[:2000] + "\n\n[dim]... (truncated, full text available)[/dim]"
    else:
        doc_display = doc_text

    console.print()
    console.print(Panel(doc_display, title="[bold green]Document[/bold green]"))

    # Relevance scale reference
    console.print()
    table = Table(title="Relevance Scale", show_header=True, header_style="bold")
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Description")
    for score, desc in RELEVANCE_LABELS.items():
        table.add_row(str(score), desc)
    console.print(table)

    if not blind:
        # After review, show the disagreement
        console.print()
        console.print(
            Panel(
                f"[yellow]Human Label:[/yellow] {item['gold_label']} ({RELEVANCE_LABELS[item['gold_label']].split(' - ')[0]})\n"
                f"[yellow]Predicted Label:[/yellow] {item['predicted_label']} ({RELEVANCE_LABELS[item['predicted_label']].split(' - ')[0]})\n"
                f"[yellow]Disagreement:[/yellow] {item['disagreement_level']} levels",
                title="[bold yellow]Original Labels (Revealed)[/bold yellow]",
            )
        )


def get_existing_review(session_data: dict, sample_id: str) -> Optional[dict]:
    """Check if a sample was already reviewed."""
    for review in session_data["reviews"]:
        if review["sample_id"] == sample_id:
            return review
    return None


def update_or_add_review(session_data: dict, review: dict):
    """Update existing review or add new one."""
    for i, r in enumerate(session_data["reviews"]):
        if r["sample_id"] == review["sample_id"]:
            session_data["reviews"][i] = review
            return
    session_data["reviews"].append(review)


def run_review_session(session_file: Path):
    """Run the interactive review session."""
    session_data = load_session(session_file)
    disagreements = session_data["disagreements"]
    current_index = session_data["current_index"]
    total = len(disagreements)

    console.print(f"\n[bold]Session loaded:[/bold] {session_file.name}")
    console.print(f"Total hard disagreements: {total}")
    console.print(f"Already reviewed: {len(session_data['reviews'])}")
    console.print(f"Starting at: {current_index + 1}\n")

    if current_index >= total:
        console.print("[green]All samples have been reviewed![/green]")
        export_results(session_data, session_file)
        return

    while current_index < total:
        item = disagreements[current_index]
        existing_review = get_existing_review(session_data, item["sample_id"])

        # Display sample (blind mode first)
        display_sample_for_review(item, current_index, total, blind=True)

        # Check if already reviewed
        if existing_review:
            console.print(
                f"\n[dim]Previously reviewed: Label={existing_review['user_label']}, "
                f"Notes: {existing_review.get('notes', 'None')}[/dim]"
            )

        # Get user input
        console.print("\n[bold]Commands:[/bold]")
        console.print("  0-3: Enter your relevance judgement")
        console.print("  b: Go back to previous sample")
        console.print("  s: Skip this sample")
        console.print("  f: Show full document text")
        console.print("  r: Reveal original labels (ends blind review for this sample)")
        console.print("  q: Save and quit")
        console.print()

        action = Prompt.ask("Your action", default="r" if existing_review else "")

        if action == "q":
            session_data["current_index"] = current_index
            save_session(session_file, session_data)
            console.print(f"\n[green]Session saved to {session_file}[/green]")
            console.print(f"Progress: {len(session_data['reviews'])}/{total} reviewed")
            export_results(session_data, session_file)
            return

        elif action == "b":
            if current_index > 0:
                current_index -= 1
            else:
                console.print("[yellow]Already at first sample[/yellow]")
                Prompt.ask("Press Enter to continue")
            continue

        elif action == "s":
            current_index += 1
            continue

        elif action == "f":
            console.print(Panel(item["doc_text"], title="[bold]Full Document[/bold]"))
            Prompt.ask("Press Enter to continue")
            continue

        elif action == "r":
            # Reveal and then ask for judgement
            display_sample_for_review(item, current_index, total, blind=False)
            console.print()

            # Now get the judgement
            while True:
                try:
                    user_label = IntPrompt.ask(
                        "Your relevance judgement (0-3)",
                        default=existing_review["user_label"] if existing_review else None,
                    )
                    if 0 <= user_label <= 3:
                        break
                    console.print("[red]Please enter 0, 1, 2, or 3[/red]")
                except (ValueError, TypeError):
                    console.print("[red]Please enter a number 0-3[/red]")

            notes = Prompt.ask(
                "Notes (optional)",
                default=existing_review.get("notes", "") if existing_review else "",
            )

            # Determine who was right
            gold = item["gold_label"]
            predicted = item["predicted_label"]

            if user_label == gold:
                assessment = "human_correct"
            elif user_label == predicted:
                assessment = "model_correct"
            elif abs(user_label - gold) < abs(user_label - predicted):
                assessment = "human_closer"
            elif abs(user_label - predicted) < abs(user_label - gold):
                assessment = "model_closer"
            else:
                assessment = "neither"

            review = {
                "sample_id": item["sample_id"],
                "sequence_number": item["sequence_number"],
                "gold_label": gold,
                "predicted_label": predicted,
                "user_label": user_label,
                "assessment": assessment,
                "notes": notes if notes else None,
                "reviewed_at": datetime.now().isoformat(),
            }

            update_or_add_review(session_data, review)
            session_data["current_index"] = current_index + 1
            save_session(session_file, session_data)

            current_index += 1

        elif action in ["0", "1", "2", "3"]:
            # Direct judgement without revealing
            user_label = int(action)
            notes = Prompt.ask("Notes (optional)", default="")

            # Now reveal to show assessment
            display_sample_for_review(item, current_index, total, blind=False)

            gold = item["gold_label"]
            predicted = item["predicted_label"]

            if user_label == gold:
                assessment = "human_correct"
                console.print("[green]You agreed with the human rater![/green]")
            elif user_label == predicted:
                assessment = "model_correct"
                console.print("[blue]You agreed with the model prediction![/blue]")
            elif abs(user_label - gold) < abs(user_label - predicted):
                assessment = "human_closer"
                console.print("[yellow]Human was closer to your judgement[/yellow]")
            elif abs(user_label - predicted) < abs(user_label - gold):
                assessment = "model_closer"
                console.print("[yellow]Model was closer to your judgement[/yellow]")
            else:
                assessment = "neither"
                console.print("[dim]Neither was clearly closer[/dim]")

            review = {
                "sample_id": item["sample_id"],
                "sequence_number": item["sequence_number"],
                "gold_label": gold,
                "predicted_label": predicted,
                "user_label": user_label,
                "assessment": assessment,
                "notes": notes if notes else None,
                "reviewed_at": datetime.now().isoformat(),
            }

            update_or_add_review(session_data, review)
            session_data["current_index"] = current_index + 1
            save_session(session_file, session_data)

            Prompt.ask("\nPress Enter to continue")
            current_index += 1

        else:
            console.print("[red]Invalid input. Please try again.[/red]")
            Prompt.ask("Press Enter to continue")

    # All done
    console.print("\n[bold green]Review complete![/bold green]")
    export_results(session_data, session_file)


def export_results(session_data: dict, session_file: Path):
    """Export review results summary."""
    reviews = session_data["reviews"]
    if not reviews:
        console.print("[yellow]No reviews to export yet.[/yellow]")
        return

    # Summary statistics
    assessments = [r["assessment"] for r in reviews]
    human_correct = assessments.count("human_correct")
    model_correct = assessments.count("model_correct")
    human_closer = assessments.count("human_closer")
    model_closer = assessments.count("model_closer")
    neither = assessments.count("neither")
    total = len(reviews)

    console.print("\n[bold]Review Summary:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Assessment")
    table.add_column("Count", justify="right")
    table.add_column("Percentage", justify="right")

    table.add_row("Human Correct", str(human_correct), f"{100*human_correct/total:.1f}%")
    table.add_row("Model Correct", str(model_correct), f"{100*model_correct/total:.1f}%")
    table.add_row("Human Closer", str(human_closer), f"{100*human_closer/total:.1f}%")
    table.add_row("Model Closer", str(model_closer), f"{100*model_closer/total:.1f}%")
    table.add_row("Neither", str(neither), f"{100*neither/total:.1f}%")
    table.add_row("[bold]Total Reviewed[/bold]", f"[bold]{total}[/bold]", "100%")
    console.print(table)

    # Export to CSV for potential publication
    csv_file = session_file.with_suffix(".csv")
    with open(csv_file, "w") as f:
        f.write("sample_id,sequence_number,gold_label,predicted_label,user_label,assessment,notes\n")
        for r in reviews:
            notes = (r.get("notes") or "").replace('"', '""')
            f.write(
                f'{r["sample_id"]},{r["sequence_number"]},{r["gold_label"]},'
                f'{r["predicted_label"]},{r["user_label"]},{r["assessment"]},"{notes}"\n'
            )
    console.print(f"\n[green]Results exported to:[/green] {csv_file}")


@app.command()
def main(
    run_name: Optional[str] = typer.Option(
        None, "--run-name", "-r", help="Name of the run to analyze"
    ),
    run_type: Optional[str] = typer.Option(
        None,
        "--run-type",
        "-t",
        help="Type of run: 'infer' (single model) or 'aggregate' (ensemble)",
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Path to session file to resume"
    ),
    list_sessions: bool = typer.Option(
        False, "--list", "-l", help="List existing review sessions"
    ),
):
    """Interactive review of hard disagreements between LLM and human judgements.

    Hard disagreements are cases where the predicted label differs from the
    human annotation by 2 or more levels on the 0-3 relevance scale.
    """
    sessions_dir = Path(__file__).parent.parent / "artifacts" / "review_sessions"

    if list_sessions:
        if not sessions_dir.exists():
            console.print("[yellow]No review sessions found.[/yellow]")
            return

        sessions = sorted(sessions_dir.glob("session_*.json"))
        if not sessions:
            console.print("[yellow]No review sessions found.[/yellow]")
            return

        console.print("\n[bold]Existing Review Sessions:[/bold]\n")
        for s in sessions:
            data = load_session(s)
            reviewed = len(data["reviews"])
            total = data["total_disagreements"]
            console.print(
                f"  {s.name}: {reviewed}/{total} reviewed "
                f"({data['run_name']}, {data['run_type']})"
            )
        console.print(f"\nResume with: python {__file__} --resume <session_file>")
        return

    if resume:
        if not resume.exists():
            # Try relative to sessions dir
            resume = sessions_dir / resume.name
        if not resume.exists():
            console.print(f"[red]Session file not found: {resume}[/red]")
            raise typer.Exit(1)
        run_review_session(resume)
        return

    if not run_name or not run_type:
        console.print("[red]Please specify --run-name and --run-type[/red]")
        console.print("\nExamples:")
        console.print(
            '  python scripts/review_hard_disagreements.py -r "-ensemble-1-to-5-majority-vote-average" -t aggregate'
        )
        console.print(
            '  python scripts/review_hard_disagreements.py -r "gpt-5-1-all-samples-base-reference" -t infer'
        )
        console.print("\nOr resume an existing session:")
        console.print("  python scripts/review_hard_disagreements.py --list")
        raise typer.Exit(1)

    if run_type not in ["infer", "aggregate"]:
        console.print("[red]run-type must be 'infer' or 'aggregate'[/red]")
        raise typer.Exit(1)

    # Load disagreements
    console.print(f"\n[bold]Loading hard disagreements from {run_type} run: {run_name}[/bold]")

    try:
        if run_type == "infer":
            disagreements = load_hard_disagreements_from_infer(run_name)
        else:
            disagreements = load_hard_disagreements_from_aggregate(run_name)
    except Exception as e:
        console.print(f"[red]Error loading data: {e}[/red]")
        raise typer.Exit(1)

    if not disagreements:
        console.print("[green]No hard disagreements found![/green]")
        return

    # Sort by disagreement level (worst first) then by sequence number
    disagreements.sort(key=lambda x: (-x["disagreement_level"], x["sequence_number"]))

    console.print(f"Found [bold]{len(disagreements)}[/bold] hard disagreements")

    # Show distribution
    level_2 = sum(1 for d in disagreements if d["disagreement_level"] == 2)
    level_3 = sum(1 for d in disagreements if d["disagreement_level"] == 3)
    console.print(f"  - 2-level disagreements: {level_2}")
    console.print(f"  - 3-level disagreements: {level_3}")

    if not Confirm.ask("\nStart review session?"):
        return

    # Create session file
    session_file = create_session_file(run_name, run_type, disagreements)
    console.print(f"\n[green]Session created: {session_file}[/green]")

    run_review_session(session_file)


if __name__ == "__main__":
    app()
