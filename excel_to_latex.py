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
    "openrouter link",
    "other links",
    "note",
    "Output modalities",
    "release",
    "Cost",
    "Ctx",
    "Reasoning"
]

# Columns that should wrap text with specified width (in cm)
WRAPPED_COLUMNS = {
    "Use Case" : "7cm",
    "Model" : "2cm",
    "Size" : "1cm",
    "Training Data" : "7cm",
    "Input" : "3cm",
    # "Column Name": "3cm",
    # "Long Description": "5cm",
}

# Columns to apply specific transformations
COLUMN_TRANSFORMS = {
    "Model": lambda x: x.split("/")[-1] if isinstance(x, str) and "/" in x else x,
    "Cost": lambda x: round(x, 3) if pd.notna(x) and isinstance(x, (int, float)) else x,
}


def excel_to_latex(
    excel_path: str,
    output_path: str | None = None,
    sheet_name: str | int = 0,
    **latex_kwargs
) -> str:
    """
    Convert Excel file to LaTeX table with formatting.

    Features:
    - Excludes columns listed in EXCLUDED_COLUMNS
    - Applies transformations from COLUMN_TRANSFORMS (e.g., strip provider from Model)
    - Makes all headers bold
    - Supports column wrapping via WRAPPED_COLUMNS (e.g., {"Column": "3cm"})

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

    # Apply column transformations
    for col, transform in COLUMN_TRANSFORMS.items():
        if col in df.columns:
            df[col] = df[col].apply(transform)

    # Build column format with wrapped columns
    if 'column_format' not in latex_kwargs or latex_kwargs['column_format'] is None:
        col_format = []
        for col in df.columns:
            if col in WRAPPED_COLUMNS:
                col_format.append(f"p{{{WRAPPED_COLUMNS[col]}}}")
            else:
                col_format.append("l")
        column_format_str = "".join(col_format)
    else:
        column_format_str = latex_kwargs['column_format']

    # Make headers bold by temporarily renaming columns
    original_columns = df.columns.tolist()
    df.columns = [f"\\textbf{{{col}}}" for col in df.columns]

    # Convert to LaTeX with sensible defaults
    # Note: pandas 2.x always uses booktabs style (requires \usepackage{booktabs})
    latex_defaults = {
        'index': False,
        'escape': False,  # Disable escape to allow \textbf{} in headers
        'column_format': column_format_str,
        'longtable': False,
        'caption': None,
        'label': None,
    }
    latex_defaults.update(latex_kwargs)

    latex_str = df.to_latex(**latex_defaults)

    # Restore original column names
    df.columns = original_columns

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
