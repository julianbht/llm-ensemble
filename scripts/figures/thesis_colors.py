"""
BHT Thesis Style Configuration

Official BHT Berlin colors and matplotlib styling for thesis figures.
Import this module in all plotting scripts for consistent theming.

Usage:
    from thesis_colors import BHT_COLORS, ENSEMBLE_PALETTE, FONTSIZE, FONTSIZE_SMALL, FIGURE_WIDTH, apply_thesis_style

    apply_thesis_style()  # Call once at start of script
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 4))
    plt.bar(..., color=BHT_COLORS["blue"])
"""

import matplotlib.pyplot as plt

# =============================================================================
# Figure dimensions (matches LaTeX \linewidth of 426.79pt)
# =============================================================================
FIGURE_WIDTH = 5.9  # inches (426.79pt / 72.27)

# =============================================================================
# Font sizes
# =============================================================================
FONTSIZE = 9         # Standard size for most text (title, axis labels, ticks, legend)
FONTSIZE_SMALL = 7   # Smaller annotations (e.g., value labels on bars)
FONTSIZE_XSMALL = 6  # Very small annotations (e.g., dense bar charts)


def apply_thesis_style():
    """Apply thesis-consistent matplotlib style. Call once at start of script."""
    plt.rcParams.update(
        {
            # Font family: Latin Modern Roman with fallbacks
            "font.family": "serif",
            "font.serif": ["Latin Modern Roman", "Times New Roman", "DejaVu Serif"],
            # Mathtext with Computer Modern for numerals
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "mathtext.it": "serif:italic",
            "mathtext.bf": "serif:bold",
            # No LaTeX required
            "text.usetex": False,
            # Font sizes
            "font.size": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "axes.titlesize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            # Misc
            "axes.unicode_minus": False,
            # Axis lines
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.pad": 6,
            "ytick.major.pad": 6,
            "figure.constrained_layout.use": True,
        }
    )

# Primary BHT colors (converted from LaTeX \xdefinecolor RGB definitions)
BHT_COLORS = {
    "gray": "#555555",  # bhtGray (0.333, 0.333, 0.333) - base gray
    "turquoise": "#00A0AA",  # bhtTurquoise (0, 0.627, 0.666)
    "cyan": "#00A0AA",  # bhtCyan (same as turquoise)
    "yellow": "#FFC900",  # bhtYellow (1, 0.788, 0)
    "red": "#EA3B06",  # bhtRed (0.918, 0.231, 0.025)
    "blue": "#004282",  # bhtBlue (0, 0.259, 0.510)
}

# Gray scale variants (light to dark)
GRAY_SCALE = {
    "very_light": "#EEEEEE",
    "light": "#BBBBBB",
    "medium": "#888888",
    "base": "#555555",
    "dark": "#3B3B3B",
    "very_dark": "#222222",
}

# Canonical color ordering for visual aesthetics
# Use this when you want a specific pleasing color progression
# Order: Yellow → Red → Blue → Turquoise → Grayscale
CANONICAL_ORDER = [
    BHT_COLORS["yellow"],
    BHT_COLORS["red"],
    BHT_COLORS["blue"],
    BHT_COLORS["turquoise"],
    GRAY_SCALE["medium"],
]

# Sequential palette for ensemble member models (5 distinct colors)
# Uses CANONICAL_ORDER for visual consistency
ENSEMBLE_PALETTE = CANONICAL_ORDER.copy()

# Alternative: gray gradient palette (if you want neutral colors)
ENSEMBLE_PALETTE_GRAY = [
    GRAY_SCALE["very_light"],
    GRAY_SCALE["light"],
    GRAY_SCALE["medium"],
    GRAY_SCALE["base"],
    GRAY_SCALE["dark"],
]

# Reference model color (for single baseline models)
# Dark gray is the default for reference/baseline models
REFERENCE_COLOR = GRAY_SCALE["dark"]  # #3B3B3B
