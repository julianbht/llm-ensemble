#!/usr/bin/env python3
"""
Generate LaTeX table comparing Cohen's Kappa and Krippendorff's Alpha rankings
from the TREC LLM Judge Challenge results.
"""

import pandas as pd
from pathlib import Path

# Configuration
INPUT_FILE = Path("artifacts/tables/judge-challenge-ranking.xlsx")
OUTPUT_FILE = Path("artifacts/tables/judge-challenge-rankings-latex.tex")
SHEET_NAME = "Sheet1"
SUBMISSION_COL = "submission-id"
COHENS_KAPPA_COL = "cohenskappa"
KRIPPENDORFF_ALPHA_COL = "krippendorfalpha"

# Submissions to highlight (add your submission IDs here)
# Note: Submission IDs are case-sensitive and must match exactly as they appear in the Excel file
HIGHLIGHT_SUBMISSIONS = [
    "GPT 5.1",  # Appears at rank 13 (Cohen's Kappa), rank 8 (Krippendorff's Alpha)
    "Ensemble (MVA)",
    "Ensemble (MVR)",
    "Ensemble (AV)",
    # Add more submission IDs to highlight here
]
HIGHLIGHT_COLOR = (
    "bhtVeryLightGray"  # LaTeX color name (requires \xdefinecolor or \definecolor)
)

# Table metadata
CAPTION = "Comparison of submissions ranked by Cohen's Kappa and Krippendorff's Alpha agreement metrics."
LABEL = "tab:judge-challenge-rankings"


def load_data(filepath: Path, sheet_name: str) -> pd.DataFrame:
    """Load data from Excel file."""
    df = pd.read_excel(filepath, sheet_name=sheet_name)
    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table with side-by-side rankings."""
    # Sort by Cohen's Kappa (descending)
    cohens_ranked = df.sort_values(COHENS_KAPPA_COL, ascending=False).reset_index(
        drop=True
    )
    cohens_ranked["rank"] = range(1, len(cohens_ranked) + 1)

    # Sort by Krippendorff's Alpha (descending)
    kripp_ranked = df.sort_values(KRIPPENDORFF_ALPHA_COL, ascending=False).reset_index(
        drop=True
    )
    kripp_ranked["rank"] = range(1, len(kripp_ranked) + 1)

    # Start LaTeX table
    lines = [
        "% Requires: \\usepackage[table]{xcolor}",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        f"\\caption{{{CAPTION}}}",
        f"\\label{{{LABEL}}}",
        "\\begin{tabular}{rlc|rlc}",
        "\\toprule",
        "\\multicolumn{3}{c|}{\\textbf{Ranked by Cohen's Kappa}} & \\multicolumn{3}{c}{\\textbf{Ranked by Krippendorff's Alpha}} \\\\",
        "\\textbf{Rank} & \\textbf{Submission} & \\textbf{$\\kappa$} & \\textbf{Rank} & \\textbf{Submission} & \\textbf{$\\alpha$} \\\\",
        "\\midrule",
    ]

    # Add data rows
    for i in range(len(df)):
        cohens_row = cohens_ranked.iloc[i]
        kripp_row = kripp_ranked.iloc[i]

        # Check if each submission should be highlighted
        cohens_id = cohens_row[SUBMISSION_COL]
        kripp_id = kripp_row[SUBMISSION_COL]
        highlight_cohens = cohens_id in HIGHLIGHT_SUBMISSIONS
        highlight_kripp = kripp_id in HIGHLIGHT_SUBMISSIONS

        # Build cells with conditional cell coloring
        cohens_rank = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{cohens_row['rank']}"
            if highlight_cohens
            else str(cohens_row["rank"])
        )
        cohens_name = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{cohens_id}"
            if highlight_cohens
            else cohens_id
        )
        cohens_value = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{cohens_row[COHENS_KAPPA_COL]:.4f}"
            if highlight_cohens
            else f"{cohens_row[COHENS_KAPPA_COL]:.4f}"
        )

        kripp_rank = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{kripp_row['rank']}"
            if highlight_kripp
            else str(kripp_row["rank"])
        )
        kripp_name = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{kripp_id}"
            if highlight_kripp
            else kripp_id
        )
        kripp_value = (
            f"\\cellcolor{{{HIGHLIGHT_COLOR}}}{kripp_row[KRIPPENDORFF_ALPHA_COL]:.4f}"
            if highlight_kripp
            else f"{kripp_row[KRIPPENDORFF_ALPHA_COL]:.4f}"
        )

        line = (
            f"{cohens_rank} & "
            f"{cohens_name} & "
            f"{cohens_value} & "
            f"{kripp_rank} & "
            f"{kripp_name} & "
            f"{kripp_value} \\\\"
        )
        lines.append(line)

    # Close table
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def main():
    """Main execution."""
    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    input_path = project_root / INPUT_FILE
    output_path = project_root / OUTPUT_FILE

    print(f"Reading data from: {input_path}")
    df = load_data(input_path, SHEET_NAME)

    print(f"Loaded {len(df)} submissions")
    print(f"Columns: {list(df.columns)}")

    # Generate LaTeX
    latex_table = generate_latex_table(df)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_table)

    print(f"\nLaTeX table written to: {output_path}")
    print("\n--- Generated LaTeX ---")
    print(latex_table)


if __name__ == "__main__":
    main()
