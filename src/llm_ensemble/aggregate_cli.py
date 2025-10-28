"""Aggregate CLI - Combine judgements using ensemble strategies.

TODO: Implement majority vote, weighted vote, and partial relevance aggregation.
"""

from llm_ensemble.libs.runtime.env import load_runtime_config
from llm_ensemble.libs.logging.logger import get_logger

# Load runtime configuration early
load_runtime_config()

logger = get_logger("aggregate")
logger.info("Aggregate CLI placeholder - implement majority/weighted vote, partials")
