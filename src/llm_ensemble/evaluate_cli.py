"""Evaluate CLI - Compute metrics and generate HTML reports.

TODO: Implement metrics computation and HTML report generation with reproducibility footers.
"""

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.logging.logger import get_logger

# Load runtime configuration early
load_runtime_config()

logger = get_logger("evaluate")
logger.info("Evaluate CLI placeholder - implement metrics computation and report.html generation")
