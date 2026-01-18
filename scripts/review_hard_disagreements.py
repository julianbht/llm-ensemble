#!/usr/bin/env python3
"""Interactive review tool for hard disagreements between LLM and human judgements.

Usage:
    python scripts/review_hard_disagreements.py -r "-ensemble-1-to-5-majority-vote-average" -t aggregate
    python scripts/review_hard_disagreements.py -r "gpt-5-1-all-samples-base-reference" -t infer
    python scripts/review_hard_disagreements.py --resume session_file.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load runtime env configuration (DATABASE_URL, API keys, etc.)
from llm_ensemble.libs.runtime.env import load_runtime_config
load_runtime_config()

from llm_ensemble.libs.db.base import get_engine
from llm_ensemble.libs.db.session import session_context
from llm_ensemble.infer.adapters.driven.io.db.orms import InferRunORM, LLMJudgementORM
from llm_ensemble.aggregate.adapters.driven.io.orms import (
    AggregateRunORM,
    AggregatedDatasetORM,
    AggregatedVoteORM,
    AggregationVoteORM,
)
from llm_ensemble.ingest.adapters.driven.io.db.orms import (
    NormalizedDatasetJudgingSampleORM,
    JudgingSampleORM,
    QueryORM,
    DocumentORM,
)
from sqlalchemy.orm import selectinload, joinedload


app = typer.Typer(pretty_exceptions_enable=False)
console = Console()


def load_hard_disagreements_from_infer(run_name: str) -> list[dict]:
    """Load hard disagreements from an infer run (single model)."""
    engine = get_engine()
    disagreements = []

    with session_context(engine) as session:
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

        judgements = (
            session.query(LLMJudgementORM)
            .filter_by(infer_run_output_id=infer_run_output.id)
            .options(joinedload(LLMJudgementORM.llm_score))
            .order_by(LLMJudgementORM.normalized_dataset_judging_sample_id)
            .all()
        )

        for j in judgements:
            predicted = j.llm_score.label if j.llm_score else None
            if predicted is None:
                continue

            sample = (
                session.query(
                    NormalizedDatasetJudgingSampleORM,
                    JudgingSampleORM,
                    QueryORM,
                    DocumentORM,
                )
                .join(JudgingSampleORM, NormalizedDatasetJudgingSampleORM.judging_sample_id == JudgingSampleORM.id)
                .join(QueryORM, JudgingSampleORM.query_id == QueryORM.id)
                .join(DocumentORM, JudgingSampleORM.document_id == DocumentORM.id)
                .filter(NormalizedDatasetJudgingSampleORM.id == j.normalized_dataset_judging_sample_id)
                .first()
            )

            if not sample:
                continue

            ds_orm, js_orm, query_orm, doc_orm = sample
            gold = js_orm.gold_score

            if abs(int(predicted) - int(gold)) >= 2:
                disagreements.append({
                    "sample_id": str(ds_orm.id),
                    "sequence_number": ds_orm.sequence_number,
                    "query_text": query_orm.query_text,
                    "doc_text": doc_orm.doc_text,
                    "gold_label": int(gold),
                    "predicted_label": int(predicted),
                    "disagreement_level": abs(int(predicted) - int(gold)),
                })

    return disagreements


def load_hard_disagreements_from_aggregate(run_name: str) -> list[dict]:
    """Load hard disagreements from an aggregate run (ensemble)."""
    engine = get_engine()
    disagreements = []

    with session_context(engine) as session:
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

        for vote in aggregated_dataset.aggregated_votes:
            predicted = vote.final_label
            if predicted is None:
                continue

            agg_vote = (
                session.query(AggregatedVoteORM)
                .filter_by(id=vote.id)
                .options(
                    selectinload(AggregatedVoteORM.aggregation_votes).joinedload(
                        AggregationVoteORM.llm_judgement
                    )
                )
                .first()
            )

            if not agg_vote or not agg_vote.aggregation_votes:
                continue

            sample_id = agg_vote.aggregation_votes[0].llm_judgement.normalized_dataset_judging_sample_id

            sample = (
                session.query(
                    NormalizedDatasetJudgingSampleORM,
                    JudgingSampleORM,
                    QueryORM,
                    DocumentORM,
                )
                .join(JudgingSampleORM, NormalizedDatasetJudgingSampleORM.judging_sample_id == JudgingSampleORM.id)
                .join(QueryORM, JudgingSampleORM.query_id == QueryORM.id)
                .join(DocumentORM, JudgingSampleORM.document_id == DocumentORM.id)
                .filter(NormalizedDatasetJudgingSampleORM.id == sample_id)
                .first()
            )

            if not sample:
                continue

            ds_orm, js_orm, query_orm, doc_orm = sample
            gold = js_orm.gold_score

            if abs(int(predicted) - int(gold)) >= 2:
                disagreements.append({
                    "sample_id": str(ds_orm.id),
                    "sequence_number": ds_orm.sequence_number,
                    "query_text": query_orm.query_text,
                    "doc_text": doc_orm.doc_text,
                    "gold_label": int(gold),
                    "predicted_label": int(predicted),
                    "disagreement_level": abs(int(predicted) - int(gold)),
                })

    return disagreements


def create_session_file(run_name: str, run_type: str, disagreements: list[dict]) -> Path:
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
        "reviews": [],
    }

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    return session_file


def load_session(session_file: Path) -> dict:
    with open(session_file) as f:
        return json.load(f)


def save_session(session_file: Path, session_data: dict):
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)


def get_existing_review(session_data: dict, sample_id: str) -> Optional[dict]:
    for review in session_data["reviews"]:
        if review["sample_id"] == sample_id:
            return review
    return None


def update_or_add_review(session_data: dict, review: dict):
    for i, r in enumerate(session_data["reviews"]):
        if r["sample_id"] == review["sample_id"]:
            session_data["reviews"][i] = review
            return
    session_data["reviews"].append(review)


def run_review_session(session_file: Path):
    session_data = load_session(session_file)
    disagreements = session_data["disagreements"]
    current_index = session_data["current_index"]
    total = len(disagreements)

    console.print(f"[bold]Session:[/bold] {session_file.name}")
    console.print(f"Hard disagreements: {total}, Reviewed: {len(session_data['reviews'])}")

    while current_index < total:
        item = disagreements[current_index]
        existing = get_existing_review(session_data, item["sample_id"])

        console.clear()
        console.print(f"[bold blue]Sample {current_index + 1}/{total}[/bold blue] (seq: {item['sequence_number']}, diff: {item['disagreement_level']})\n")

        console.print(Panel(item["query_text"], title="[cyan]Query[/cyan]"))

        doc = item["doc_text"]
        if len(doc) > 3000:
            doc = doc[:3000] + "\n\n[dim]...(truncated)[/dim]"
        console.print(Panel(doc, title="[green]Document[/green]"))

        if existing:
            console.print(f"[dim]Previously: {existing['user_label']}[/dim]")

        console.print("\n[bold]0-3[/bold]=judge  [bold]b[/bold]=back  [bold]s[/bold]=skip  [bold]f[/bold]=full doc  [bold]q[/bold]=quit")
        action = Prompt.ask("", default="")

        if action == "q":
            session_data["current_index"] = current_index
            save_session(session_file, session_data)
            console.print(f"\n[green]Saved. Progress: {len(session_data['reviews'])}/{total}[/green]")
            export_results(session_data, session_file)
            return

        elif action == "b":
            if current_index > 0:
                current_index -= 1
            continue

        elif action == "s":
            current_index += 1
            continue

        elif action == "f":
            console.print(Panel(item["doc_text"], title="[bold]Full Document[/bold]"))
            Prompt.ask("Enter to continue")
            continue

        elif action in ["0", "1", "2", "3"]:
            user_label = int(action)
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
                "reviewed_at": datetime.now().isoformat(),
            }

            update_or_add_review(session_data, review)
            session_data["current_index"] = current_index + 1
            save_session(session_file, session_data)
            current_index += 1

    console.print("\n[bold green]All done![/bold green]")
    export_results(session_data, session_file)


def export_results(session_data: dict, session_file: Path):
    reviews = session_data["reviews"]
    if not reviews:
        return

    assessments = [r["assessment"] for r in reviews]
    total = len(reviews)

    console.print("\n[bold]Summary:[/bold]")
    for a in ["human_correct", "model_correct", "human_closer", "model_closer", "neither"]:
        count = assessments.count(a)
        console.print(f"  {a}: {count} ({100*count/total:.0f}%)")

    csv_file = session_file.with_suffix(".csv")
    with open(csv_file, "w") as f:
        f.write("sample_id,sequence_number,gold_label,predicted_label,user_label,assessment\n")
        for r in reviews:
            f.write(f'{r["sample_id"]},{r["sequence_number"]},{r["gold_label"]},{r["predicted_label"]},{r["user_label"]},{r["assessment"]}\n')
    console.print(f"[green]Exported: {csv_file}[/green]")


@app.command()
def main(
    run_name: Optional[str] = typer.Option(None, "--run-name", "-r"),
    run_type: Optional[str] = typer.Option(None, "--run-type", "-t"),
    resume: Optional[Path] = typer.Option(None, "--resume"),
    list_sessions: bool = typer.Option(False, "--list", "-l"),
):
    """Review hard disagreements (|predicted - gold| >= 2)."""
    sessions_dir = Path(__file__).parent.parent / "artifacts" / "review_sessions"

    if list_sessions:
        if not sessions_dir.exists():
            console.print("No sessions.")
            return
        for s in sorted(sessions_dir.glob("session_*.json")):
            data = load_session(s)
            console.print(f"  {s.name}: {len(data['reviews'])}/{data['total_disagreements']} ({data['run_name']})")
        return

    if resume:
        if not resume.exists():
            resume = sessions_dir / resume.name
        run_review_session(resume)
        return

    if not run_name or not run_type:
        console.print("[red]Need --run-name and --run-type[/red]")
        raise typer.Exit(1)

    console.print(f"Loading from {run_type}: {run_name}...")

    if run_type == "infer":
        disagreements = load_hard_disagreements_from_infer(run_name)
    elif run_type == "aggregate":
        disagreements = load_hard_disagreements_from_aggregate(run_name)
    else:
        console.print("[red]run-type must be 'infer' or 'aggregate'[/red]")
        raise typer.Exit(1)

    if not disagreements:
        console.print("[green]No hard disagreements![/green]")
        return

    # Sort by disagreement level (worst first) then sequence number
    disagreements.sort(key=lambda x: (-x["disagreement_level"], x["sequence_number"]))

    level_2 = sum(1 for d in disagreements if d["disagreement_level"] == 2)
    level_3 = sum(1 for d in disagreements if d["disagreement_level"] == 3)
    console.print(f"Found {len(disagreements)} hard disagreements (2-level: {level_2}, 3-level: {level_3})")

    session_file = create_session_file(run_name, run_type, disagreements)
    run_review_session(session_file)


if __name__ == "__main__":
    app()
