"""
BHT Thesis Color Scheme

Official BHT Berlin colors converted from LaTeX RGB to hex format.
Import this module in all plotting scripts for consistent theming.

Usage:
    from thesis_colors import BHT_COLORS, ENSEMBLE_PALETTE

    plt.bar(..., color=BHT_COLORS["blue"])
    # or for multiple colors:
    plt.bar(..., color=ENSEMBLE_PALETTE[0])
"""

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
