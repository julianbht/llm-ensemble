#!/usr/bin/env python3
"""Convert Excel file to LaTeX table format using pandas."""

import pandas as pd
import typer
from pathlib import Path
from typing import Optional


app = typer.Typer(
    help="Convert Excel file to LaTeX table",
    add_completion=False,
    pretty_exceptions_enable=False,  # Disable verbose tracebacks
)


EXCLUDED_COLUMNS = [
    # Add column names to exclude from LaTeX output
    # "Unnamed: 0",
    # "Notes",
    "openrouter_link",
    "other links",
    "note",
    "Output modalities",
    "release"
]


def excel_to_latex(
    excel_path: str,
    output_path: str | None = None,
    sheet_name: str | int = 0,
    **latex_kwargs
) -> str:
    """
    Convert Excel file to LaTeX table.

    Args:
        excel_path: Path to Excel file
        output_path: Optional path to write LaTeX output (if None, prints to stdout)
        sheet_name: Sheet name or index to convert (default: first sheet)
        **latex_kwargs: Additional arguments passed to DataFrame.to_latex()

    Returns:
        LaTeX table string
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Filter out excluded columns (only if they exist)
    columns_to_drop = [col for col in EXCLUDED_COLUMNS if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)

    # Convert to LaTeX with sensible defaults
    # Note: pandas 2.x always uses booktabs style (requires \usepackage{booktabs})
    latex_defaults = {
        'index': False,
        'escape': True,
        'column_format': None,
        'longtable': False,
        'caption': None,
        'label': None,
    }
    latex_defaults.update(latex_kwargs)

    latex_str = df.to_latex(**latex_defaults)

    if output_path:
        Path(output_path).write_text(latex_str)
        typer.echo(f"LaTeX table written to: {output_path}")
    else:
        typer.echo(latex_str)

    return latex_str


@app.command()
def main(
    excel_file: Path = typer.Argument(..., help="Path to Excel file", exists=True),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output LaTeX file (default: print to stdout)"),
    sheet: str = typer.Option("0", help="Sheet name or index"),
    index: bool = typer.Option(False, help="Include DataFrame index in output"),
    longtable: bool = typer.Option(False, help="Use longtable environment for multi-page tables"),
    caption: Optional[str] = typer.Option(None, help="Table caption"),
    label: Optional[str] = typer.Option(None, help="LaTeX label for referencing"),
    column_format: Optional[str] = typer.Option(None, help='LaTeX column format string (e.g., "lrr")'),
):
    """
    Convert Excel file to LaTeX table with booktabs formatting.

    Note: Output always uses booktabs style (requires \\usepackage{booktabs} in LaTeX).

    Examples:

        excel_to_latex.py model-analysis.xlsx

        excel_to_latex.py model-analysis.xlsx -o table.tex

        excel_to_latex.py model-analysis.xlsx --sheet "Sheet2"

        excel_to_latex.py model-analysis.xlsx --index --caption "My Table" --label "tab:mytable"
    """
    # Try to convert sheet to int if it looks numeric
    sheet_name: str | int = sheet
    try:
        sheet_name = int(sheet)
    except ValueError:
        pass

    latex_kwargs = {
        'index': index,
        'longtable': longtable,
    }

    if caption:
        latex_kwargs['caption'] = caption
    if label:
        latex_kwargs['label'] = label
    if column_format:
        latex_kwargs['column_format'] = column_format

    excel_to_latex(
        str(excel_file),
        str(output) if output else None,
        sheet_name=sheet_name,
        **latex_kwargs
    )


if __name__ == '__main__':
    app()
